"""MCP server for Gaggimate espresso machine."""

#!/usr/bin/env python3

import json
import asyncio
import traceback
from pathlib import Path
from typing import Optional, Union
from pydantic import ValidationError

from mcp.server.fastmcp import FastMCP
from gaggimate_mcp.config import GaggimateConfig
from gaggimate_mcp.logging_config import setup_logging, get_logger
from gaggimate_mcp.api.websocket import GaggimateWebSocketClient
from gaggimate_mcp.api.http import GaggimateHTTPClient
from gaggimate_mcp.transformers.shot import transform_shot_for_ai, VALID_DETAIL_LEVELS
from gaggimate_mcp.storage.profiles import ProfileStorage
from gaggimate_mcp.storage.ratings import RatingStorage
from gaggimate_mcp.storage.markdown import (
    MarkdownStorage,
    build_coffee_markdown,
    append_journal_entry,
    build_grind_map_markdown,
    append_grind_entry,
    build_brewing_insights_markdown,
    _sanitize_filename,
)
from gaggimate_mcp.models.rating import ShotRating, BalanceTaste
from gaggimate_mcp.errors import GaggimateError
from gaggimate_mcp.diagnostics import diagnose_connection as run_diagnostics
from gaggimate_mcp.resources import register_resources, _list_markdown_files, _read_markdown_file


# Initialize configuration and logging
config = GaggimateConfig()
setup_logging(log_level=config.log_level)
logger = get_logger(__name__)

# Create FastMCP server
mcp = FastMCP("gaggimate-mcp")

# Register MCP resources (knowledge, coffees, user data)
register_resources(mcp, config)

# Initialize clients and storage
ws_client = GaggimateWebSocketClient(config)
http_client = GaggimateHTTPClient(config)
profile_storage = ProfileStorage(config)
rating_storage = RatingStorage(config)
coffee_storage = MarkdownStorage(config.coffees_dir)
user_storage = MarkdownStorage(config.user_dir)


def _get_error_suggestion(error: GaggimateError) -> str:
    """Generate user-friendly error suggestion based on error code.

    Args:
        error: GaggimateError instance

    Returns:
        Helpful suggestion for resolving the error
    """
    from gaggimate_mcp.errors import ErrorCode

    suggestions = {
        ErrorCode.DEVICE_UNREACHABLE: (
            f"Cannot reach Gaggimate device at {config.host}. "
            "Please check: 1) Device is powered on, 2) Connected to same network, "
            "3) Correct IP address/hostname is configured. "
            "Run 'diagnose_connection' tool for detailed diagnostics."
        ),
        ErrorCode.WEBSOCKET_ERROR: (
            f"WebSocket connection failed to {config.host}. "
            "Please verify: 1) Device is online, 2) WebSocket port is accessible, "
            "3) No firewall blocking connection. "
            "Run 'diagnose_connection' tool for detailed diagnostics."
        ),
        ErrorCode.TIMEOUT: (
            f"Request timed out waiting for response from {config.host}. "
            "Please check: 1) Device is responding (try accessing web UI), "
            "2) Network connection is stable, 3) Device is not overloaded. "
            "Run 'diagnose_connection' tool for detailed diagnostics."
        ),
        ErrorCode.API_ERROR: (
            "Gaggimate API returned an error. "
            "The request format may be invalid or the device rejected the operation. "
            "Check the error message for details."
        ),
        ErrorCode.PARSE_ERROR: (
            "Failed to parse response from Gaggimate. "
            "The device may be running incompatible firmware or the data format changed. "
            "Consider updating the MCP server or checking device firmware version."
        ),
        ErrorCode.PROFILE_NOT_FOUND: (
            "Profile not found on device. "
            "Use 'manage_profile' with action='list' to see available profiles."
        ),
        ErrorCode.SHOT_NOT_FOUND: (
            "Shot not found on device. "
            "Use 'list_recent_shots' to see available shot IDs. "
            "Note: Shot IDs are 6-digit numbers (e.g., '000100')."
        ),
    }

    return suggestions.get(error.code, "Please check the error message and try again.")


@mcp.tool()
async def manage_profile(
    action: str,
    profile_id: Optional[str] = None,
    profile_name: Optional[str] = None,
    temperature: Optional[float] = None,
    phases: Optional[Union[str, list]] = None,
    confirm_delete: bool = False
) -> str:
    """Manage espresso brewing profiles on the Gaggimate espresso machine.

    Args:
        action: Action to perform:
            - 'list': List all available profiles
            - 'get': Get a specific profile by ID
            - 'create': Create a new profile (requires profile_name, temperature, phases)
            - 'update': Update an existing profile. Supports PARTIAL UPDATES - only provide
              the fields you want to change. Requires profile_id or profile_name to identify
              the profile. Omitted fields will keep their existing values.
            - 'delete': Delete an existing profile (SAFETY: Only AI-created profiles
              with ' [AI]' suffix can be deleted. Requires confirm_delete=True)
            - 'select': Select a profile as the active brewing profile. The machine will
              use this profile for the next espresso shot. Requires profile_id or profile_name.
        profile_id: Profile ID (required for 'get' and 'delete', optional for 'update')
        confirm_delete: Must be True to confirm profile deletion. This is a safety
            measure to prevent accidental deletions. Only profiles with the ' [AI]'
            suffix can be deleted by the agent.
        profile_name: Profile name. IMPORTANT: For agent-created profiles, always add ' [AI]'
            suffix (e.g., 'Ethiopian Light [AI]') so users can identify AI-created profiles.
            Design note: We use a suffix (not prefix) because profile names are displayed
            in lists on small screens - "Amizade [AI]" keeps the meaningful name visible,
            whereas "[AI] Amizade" would sort all AI profiles together alphabetically.
        temperature: Water temperature in Celsius, typically 88-96°C (required for 'create',
            optional for 'update' - omit to keep existing value)
        phases: Array of brewing phases (required for 'create', optional for 'update' -
            omit to keep existing phases). Each phase object:
            - name (str): Phase display name (e.g., 'Pre-infusion', 'Extraction', 'Decline')
            - phase (str): Phase type - 'preinfusion', 'brew', or 'decline'
            - duration (int): Maximum duration in seconds
            - valve (int): Valve setting, typically 1
            - temperature (int): Phase-specific temp offset, typically 0
            - pump (object): Pump control settings:
                - target (str): 'pressure' or 'flow'
                - pressure (float): Pressure in bar (0-12)
                - flow (float): Flow rate in ml/s
            - transition (object): How to transition into this phase:
                - type (str): 'instant', 'linear', 'ease-in', or 'ease-out'
                - duration (int): Transition duration in seconds
                - adaptive (bool): Whether transition adapts to conditions
            - targets (array, optional): Stop conditions to exit phase early:
                - type (str): 'pressure', 'flow', 'volumetric', or 'pumped'
                - operator (str): 'gte' (>=) or 'lte' (<=)
                - value (float): Threshold value

        Example phases for a classic espresso profile:
        [
            {"name": "Pre-infusion", "phase": "preinfusion", "valve": 1, "duration": 8,
             "temperature": 0, "pump": {"target": "flow", "pressure": 3, "flow": 2},
             "transition": {"type": "instant", "duration": 0, "adaptive": true}},
            {"name": "Extraction", "phase": "brew", "valve": 1, "duration": 30,
             "temperature": 0, "pump": {"target": "pressure", "pressure": 9, "flow": 0},
             "transition": {"type": "ease-in", "duration": 3, "adaptive": true},
             "targets": [{"type": "volumetric", "operator": "gte", "value": 36}]}
        ]

    Returns:
        JSON string with result
    """
    logger.info("manage_profile_called", action=action, profile_id=profile_id)

    try:
        if action == "list":
            profiles = await ws_client.list_profiles()
            return json.dumps({
                "success": True,
                "action": "list",
                "profiles": profiles,
                "count": len(profiles)
            })

        elif action == "get":
            if not profile_id:
                return json.dumps({
                    "success": False,
                    "error": "profile_id is required for 'get' action"
                })

            profile = await ws_client.load_profile(profile_id)
            if not profile:
                return json.dumps({
                    "success": False,
                    "error": f"Profile '{profile_id}' not found"
                })

            return json.dumps({
                "success": True,
                "action": "get",
                "profile": profile
            })

        elif action == "create":
            # Create requires all parameters
            if not profile_name or temperature is None or not phases:
                return json.dumps({
                    "success": False,
                    "error": "profile_name, temperature, and phases are required for create"
                })

            # Handle phases as either JSON string or already-parsed list
            if isinstance(phases, list):
                phases_list = phases
            else:
                try:
                    phases_list = json.loads(phases)
                except json.JSONDecodeError:
                    return json.dumps({
                        "success": False,
                        "error": "Invalid JSON in phases parameter"
                    })

            # Validate phases structure
            if not isinstance(phases_list, list):
                return json.dumps({
                    "success": False,
                    "error": "phases must be a JSON array"
                })

            if len(phases_list) == 0:
                return json.dumps({
                    "success": False,
                    "error": "At least one phase is required"
                })

            # Validate each phase has required fields
            for idx, phase in enumerate(phases_list):
                if not isinstance(phase, dict):
                    return json.dumps({
                        "success": False,
                        "error": f"Phase {idx} must be an object"
                    })

                # Check required fields
                required_fields = ["name", "phase", "duration"]
                missing = [f for f in required_fields if f not in phase]
                if missing:
                    return json.dumps({
                        "success": False,
                        "error": f"Phase {idx} missing required fields: {', '.join(missing)}"
                    })

                # Validate duration is a number
                if not isinstance(phase["duration"], (int, float)):
                    return json.dumps({
                        "success": False,
                        "error": f"Phase {idx} 'duration' must be a number"
                    })

            # Create new profile
            saved_profile = await ws_client.create_or_update_profile(
                label=profile_name,
                temperature=temperature,
                phases=phases_list,
                profile_id=None,
                profile_type="pro"
            )

            # Save version locally
            version_info = profile_storage.save_profile_version(
                profile_name=profile_name,
                profile_data=saved_profile,
                metadata={
                    "action": action,
                    "temperature": temperature,
                    "phase_count": len(phases_list)
                }
            )

            return json.dumps({
                "success": True,
                "action": action,
                "profile": saved_profile,
                "version_info": version_info
            })

        elif action == "update":
            # Update requires profile_id OR profile_name to identify the profile
            if not profile_id and not profile_name:
                return json.dumps({
                    "success": False,
                    "error": "profile_id or profile_name is required to identify the profile to update"
                })

            # Load existing profile first
            existing = None
            target_id = profile_id
            
            if profile_id:
                existing = await ws_client.load_profile(profile_id)
                if not existing:
                    return json.dumps({
                        "success": False,
                        "error": f"Profile with ID '{profile_id}' not found"
                    })
            else:
                # Find by name
                existing = await ws_client.find_profile_by_label(profile_name)
                if not existing:
                    return json.dumps({
                        "success": False,
                        "error": f"Profile with name '{profile_name}' not found"
                    })
                target_id = existing.get("id")

            # Use existing values as defaults, override with provided values
            final_name = profile_name if profile_name else existing.get("label")
            final_temperature = temperature if temperature is not None else existing.get("temperature")
            final_phases = None
            existing_type = existing.get("type", "pro")

            # Handle phases - use existing if not provided
            if phases:
                if isinstance(phases, list):
                    final_phases = phases
                else:
                    try:
                        final_phases = json.loads(phases)
                    except json.JSONDecodeError:
                        return json.dumps({
                            "success": False,
                            "error": "Invalid JSON in phases parameter"
                        })

                # Validate phases structure
                if not isinstance(final_phases, list):
                    return json.dumps({
                        "success": False,
                        "error": "phases must be a JSON array"
                    })

                if len(final_phases) == 0:
                    return json.dumps({
                        "success": False,
                        "error": "At least one phase is required"
                    })

                # Validate each phase has required fields
                for idx, phase in enumerate(final_phases):
                    if not isinstance(phase, dict):
                        return json.dumps({
                            "success": False,
                            "error": f"Phase {idx} must be an object"
                        })

                    required_fields = ["name", "phase", "duration"]
                    missing = [f for f in required_fields if f not in phase]
                    if missing:
                        return json.dumps({
                            "success": False,
                            "error": f"Phase {idx} missing required fields: {', '.join(missing)}"
                        })

                    if not isinstance(phase["duration"], (int, float)):
                        return json.dumps({
                            "success": False,
                            "error": f"Phase {idx} 'duration' must be a number"
                        })
            else:
                # Use existing phases
                final_phases = existing.get("phases", [])

            # Update profile
            saved_profile = await ws_client.create_or_update_profile(
                label=final_name,
                temperature=final_temperature,
                phases=final_phases,
                profile_id=target_id,
                profile_type=existing_type
            )

            # Save version locally
            version_info = profile_storage.save_profile_version(
                profile_name=final_name,
                profile_data=saved_profile,
                metadata={
                    "action": action,
                    "temperature": final_temperature,
                    "phase_count": len(final_phases)
                }
            )

            return json.dumps({
                "success": True,
                "action": action,
                "profile": saved_profile,
                "version_info": version_info
            })

        elif action == "delete":
            # Safety check 1: Require profile_id
            if not profile_id:
                return json.dumps({
                    "success": False,
                    "error": "Profile ID is required for delete action"
                })

            # Safety check 2: Require explicit confirmation
            if not confirm_delete:
                return json.dumps({
                    "success": False,
                    "error": "Delete requires confirm_delete=True. This is a safety measure. "
                             "Please confirm the user explicitly wants to delete this profile."
                })

            # Safety check 3: Load the profile and verify it has AI suffix
            ai_suffix = config.ai_profile_suffix
            existing = await ws_client.load_profile(profile_id)
            if not existing:
                return json.dumps({
                    "success": False,
                    "error": f"Profile with ID '{profile_id}' not found"
                })

            profile_label = existing.get("label", "")
            if not profile_label.endswith(ai_suffix):
                return json.dumps({
                    "success": False,
                    "error": f"Cannot delete profile '{profile_label}'. "
                             f"Only AI-created profiles (ending with '{ai_suffix}') can be deleted by the agent. "
                             "This protects user-created profiles from accidental deletion."
                })

            # All safety checks passed - delete the profile
            logger.info("deleting_profile", profile_id=profile_id, label=profile_label)
            await ws_client.delete_profile(profile_id)

            return json.dumps({
                "success": True,
                "action": "delete",
                "deleted_profile": {
                    "id": profile_id,
                    "label": profile_label
                },
                "message": f"Profile '{profile_label}' has been permanently deleted"
            })

        elif action == "select":
            # Select requires profile_id OR profile_name to identify the profile
            if not profile_id and not profile_name:
                return json.dumps({
                    "success": False,
                    "error": "profile_id or profile_name is required to identify the profile to select"
                })

            target_id = profile_id
            profile_label = None

            if profile_id:
                # Verify profile exists
                profile = await ws_client.load_profile(profile_id)
                if not profile:
                    return json.dumps({
                        "success": False,
                        "error": f"Profile with ID '{profile_id}' not found"
                    })
                profile_label = profile.get("label")
            else:
                # Find by name
                profile = await ws_client.find_profile_by_label(profile_name)
                if not profile:
                    return json.dumps({
                        "success": False,
                        "error": f"Profile with name '{profile_name}' not found"
                    })
                target_id = profile.get("id")
                profile_label = profile_name

            # Select the profile
            logger.info("selecting_profile", profile_id=target_id, label=profile_label)
            await ws_client.select_profile(target_id)

            return json.dumps({
                "success": True,
                "action": "select",
                "profile_id": target_id,
                "profile_name": profile_label,
                "message": f"Profile '{profile_label}' is now selected as the active brewing profile"
            })

        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown action '{action}'. Use: list, get, create, update, delete, select"
            })

    except GaggimateError as e:
        logger.error("manage_profile_error", action=action, error=str(e), code=e.code.value)
        return json.dumps({
            "success": False,
            "error": str(e),
            "error_code": e.code.value,
            "suggestion": _get_error_suggestion(e)
        })
    except Exception as e:
        error_msg = str(e) or f"{type(e).__name__} (no message)"
        logger.error("manage_profile_unexpected_error", action=action, error=error_msg, exception_type=type(e).__name__)
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {error_msg}",
            "exception_type": type(e).__name__
        })


@mcp.tool()
async def analyze_shot(shot_id: str, detail: str = "summary") -> str:
    """Get comprehensive shot analysis.

    Args:
        shot_id: Shot ID to analyze (will be normalized to 6 digits)
        detail: Diagnostic detail level. Options:
            - "summary" (default): Key indicators only — resistance,
              channeling risk, temperature stability, profile compliance.
              Use for quick assessments and triage.
            - "per_phase": Full diagnostics plus per-phase breakdowns
              (preinfusion/brew/decline metrics) without samples.
              Use to identify which phase has an issue.
            - "per_phase_detailed": Everything in per_phase plus ~5
              averaged samples per phase showing the curve shape.
              Use after per_phase to see pressure/flow trends.

    Returns:
        JSON string with shot analysis
    """
    # Normalize shot ID to 6 digits for consistent lookups
    normalized_id = shot_id.zfill(6)
    logger.info("analyze_shot_called", shot_id=shot_id, normalized_id=normalized_id, detail=detail)

    try:
        # Fetch shot data with normalized ID
        shot_data = await http_client.fetch_shot(normalized_id)
        if not shot_data:
            return json.dumps({
                "success": False,
                "error": f"Shot '{shot_id}' not found"
            })

        # Transform for AI analysis
        detail_level = detail if detail in VALID_DETAIL_LEVELS else "summary"
        transformed = transform_shot_for_ai(shot_data, detail=detail_level)

        # Get rating if available (using normalized ID)
        rating_data = rating_storage.get_rating(normalized_id)

        return json.dumps({
            "success": True,
            "shot": transformed,
            "rating": rating_data,
            "incomplete": shot_data.incomplete
        })

    except GaggimateError as e:
        logger.error("analyze_shot_error", shot_id=shot_id, error=str(e), code=e.code.value)
        return json.dumps({
            "success": False,
            "error": str(e),
            "error_code": e.code.value,
            "suggestion": _get_error_suggestion(e)
        })
    except Exception as e:
        error_msg = str(e) or f"{type(e).__name__} (no message)"
        tb = traceback.format_exc()
        logger.error("analyze_shot_unexpected_error", shot_id=shot_id, error=error_msg, traceback=tb)
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {error_msg}",
            "exception_type": type(e).__name__
        })


@mcp.tool()
async def manage_shot_notes(
    shot_id: str,
    action: str = "update",
    rating: Optional[int] = None,
    notes: Optional[str] = None,
    balance_taste: Optional[str] = None,
    grind_setting: Optional[str] = None,
    dose_in: Optional[float] = None,
    dose_out: Optional[float] = None,
    sync_to_device: bool = True
) -> str:
    """Manage shot notes and ratings.

    This tool syncs feedback to the Gaggimate device (via WebSocket API) and saves
    a local backup. The device is the source of truth for all shot notes.

    Args:
        shot_id: Shot ID (e.g., "100" or "000100" - will be normalized)
        action: Action to perform - "update" or "get" (default: "update")
        rating: Star rating (0-5, optional)
        notes: Tasting notes (optional)
        balance_taste: Taste balance - "bitter", "balanced", or "sour" (optional)
        grind_setting: Grinder setting used (optional)
        dose_in: Coffee dose in grams (optional)
        dose_out: Espresso output in grams (optional)
        sync_to_device: Whether to sync to Gaggimate device (default: True)

    Returns:
        JSON string with result
    """
    # Normalize shot ID - remove leading zeros for API, keep padded for local storage
    try:
        shot_id_int = int(shot_id)
        api_id = str(shot_id_int)  # For WebSocket API: "100"
        storage_id = str(shot_id_int).zfill(6)  # For local storage: "000100"
    except ValueError:
        return json.dumps({
            "success": False,
            "error": f"Invalid shot ID: '{shot_id}'. Must be a number."
        })

    logger.info("manage_shot_notes_called", shot_id=shot_id, api_id=api_id, storage_id=storage_id, action=action, rating=rating)

    try:
        if action == "get":
            # Get from device via WebSocket (device is source of truth)
            device_notes = await ws_client.get_shot_notes(api_id)
            return json.dumps({
                "success": True,
                "shot_id": api_id,
                "notes": device_notes,
                "source": "device"
            })

        elif action == "update":
            results = {
                "device_synced": False,
                "local_saved": False,
                "device_error": None
            }

            # Design note: We use a prefix for shot notes to clearly indicate
            # AI-generated content in the Gaggimate UI. This is intentionally
            # different from the suffix used for profile names because:
            # 1. Notes are free-form text where a prefix is more natural
            # 2. Profile names are displayed in lists where suffix keeps the
            #    meaningful name visible on small screens
            # Both prefixes are configurable via GAGGIMATE_AI_NOTES_PREFIX and
            # GAGGIMATE_AI_PROFILE_SUFFIX environment variables.
            agent_notes = None
            if notes:
                agent_prefix = config.ai_notes_prefix
                # Only add prefix if not already present
                if not notes.startswith(agent_prefix):
                    agent_notes = f"{agent_prefix}{notes}"
                else:
                    agent_notes = notes
            
            # Save to device if requested
            if sync_to_device:
                try:
                    await ws_client.save_shot_notes(
                        shot_id=api_id,
                        rating=rating,
                        notes=agent_notes,
                        balance_taste=balance_taste,
                        grind_setting=grind_setting,
                        dose_in=dose_in,
                        dose_out=dose_out,
                    )
                    results["device_synced"] = True
                    logger.info("shot_notes_synced_to_device", shot_id=api_id)
                except GaggimateError as e:
                    results["device_error"] = str(e)
                    logger.warning("shot_notes_device_sync_failed", shot_id=api_id, error=str(e))

            # Convert balance_taste string to enum if provided
            balance_taste_enum = None
            if balance_taste:
                try:
                    balance_taste_enum = BalanceTaste(balance_taste.lower())
                except ValueError:
                    logger.warning("invalid_balance_taste", value=balance_taste)
                    # Continue with None

            # Always save locally as backup
            shot_rating = ShotRating(
                shot_id=storage_id,
                rating=rating,
                notes=agent_notes,
                balance_taste=balance_taste_enum,
                grind_setting=grind_setting,
                dose_in=dose_in,
                dose_out=dose_out,
            )
            rating_data = rating_storage.save_rating(shot_rating)
            results["local_saved"] = True

            # Build response message
            if results["device_synced"]:
                message = "Shot notes saved to device and stored locally"
            elif results["device_error"]:
                message = f"Shot notes stored locally (device sync failed: {results['device_error']})"
            else:
                message = "Shot notes stored locally only"

            return json.dumps({
                "success": True,
                "message": message,
                "shot_id": storage_id,
                "api_id": api_id,
                "rating": rating_data,
                "sync_status": results
            })

        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown action '{action}'. Use 'update' or 'get'"
            })

    except ValidationError as e:
        # Pydantic validation error
        logger.error("manage_shot_notes_validation_error", shot_id=shot_id, error=str(e))
        return json.dumps({
            "success": False,
            "error": f"Validation error: {str(e)}"
        })
    except ValueError as e:
        # Other value errors
        logger.error("manage_shot_notes_value_error", shot_id=shot_id, error=str(e))
        return json.dumps({
            "success": False,
            "error": f"Value error: {str(e)}"
        })
    except Exception as e:
        error_msg = str(e) or f"{type(e).__name__} (no message)"
        logger.error("manage_shot_notes_unexpected_error", shot_id=shot_id, error=error_msg, exception_type=type(e).__name__)
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {error_msg}",
            "exception_type": type(e).__name__
        })


@mcp.tool()
async def diagnose_connection() -> str:
    """Run connection diagnostics to troubleshoot device connectivity issues.

    This tool checks:
    - Device reachability (ping)
    - HTTP port accessibility
    - API endpoint availability
    - HTTPS misconfiguration
    - Network latency issues

    Returns:
        JSON string with diagnostic results and troubleshooting recommendations
    """
    logger.info("diagnose_connection_called")

    try:
        results = await run_diagnostics(config)

        # Format for user-friendly output
        summary = {
            "status": results["overall_status"],
            "host": results["host"],
            "tests": results["tests"],
            "issues": results["issues"],
            "recommendations": results["recommendations"]
        }

        return json.dumps({
            "success": True,
            "diagnostics": summary
        }, indent=2)

    except Exception as e:
        logger.error("diagnose_connection_error", error=str(e))
        return json.dumps({
            "success": False,
            "error": f"Diagnostics failed: {str(e)}"
        })


@mcp.tool()
async def list_recent_shots(limit: int = 10) -> str:
    """List recent shots with optional filtering.

    Args:
        limit: Number of shots to return (default 10, max 50)

    Returns:
        JSON string with list of recent shots
    """
    logger.info("list_recent_shots_called", limit=limit)

    try:
        # Clamp limit
        limit = max(1, min(limit, 50))

        # Fetch shot index
        shots = await http_client.list_recent_shots(limit=limit)

        # Enrich with ratings
        for shot in shots:
            shot_id = shot["id"]
            rating_data = rating_storage.get_rating(shot_id)
            if rating_data:
                shot["user_rating"] = rating_data.get("rating")
                shot["user_notes"] = rating_data.get("notes")

        return json.dumps({
            "success": True,
            "shots": shots,
            "count": len(shots),
            "limit": limit
        })

    except GaggimateError as e:
        logger.error("list_recent_shots_error", limit=limit, error=str(e), code=e.code.value)
        return json.dumps({
            "success": False,
            "error": str(e),
            "error_code": e.code.value,
            "suggestion": _get_error_suggestion(e)
        })
    except Exception as e:
        error_msg = str(e) or f"{type(e).__name__} (no message)"
        logger.error("list_recent_shots_unexpected_error", limit=limit, error=error_msg, exception_type=type(e).__name__)
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {error_msg}",
            "exception_type": type(e).__name__
        })


# ── Local Data Management Tools ──────────────────────────────────────────────
# These tools manage local markdown files (coffees, user setup, grind map).
# They do NOT communicate with the Gaggimate device — only read/write files.
# Read access is via MCP resources (gaggimate:// URIs), writes are via these tools.


@mcp.tool()
async def manage_coffee(
    action: str,
    coffee_name: Optional[str] = None,
    roaster: Optional[str] = None,
    origin: Optional[str] = None,
    process: Optional[str] = None,
    variety: Optional[str] = None,
    roast_level: Optional[str] = None,
    roast_date: Optional[str] = None,
    bag_size: Optional[str] = None,
    roaster_notes: Optional[str] = None,
    freshness_note: Optional[str] = None,
    approach: Optional[str] = None,
    entry_date: Optional[str] = None,
    entry_headline: Optional[str] = None,
    entry_body: Optional[str] = None,
    content: Optional[str] = None,
) -> str:
    """Manage coffee tracking files in the coffees/ directory.

    Each coffee gets its own markdown file with bean identity, a narrative
    brewing approach, a brewing journal (dated analysis entries), and key
    insights. Raw shot numbers live on the device — this file stores
    *thinking* and *learnings*.

    Args:
        action: Action to perform:
            - 'list': List all coffee files
            - 'read': Read a specific coffee file (requires coffee_name)
            - 'create': Create a new coffee file (requires coffee_name and roaster)
            - 'log_entry': Append a brewing journal entry (analysis, not raw numbers)
            - 'update': Overwrite a coffee file with new content
            - 'delete': Delete a coffee file
        coffee_name: Coffee name (used for file identification)
        roaster: Roaster name (required for 'create')
        origin: Country/region of origin
        process: Processing method (e.g. "Washed", "Natural")
        variety: Coffee variety (e.g. "Yellow Catuai")
        roast_level: Roast level (e.g. "Medium", "Light")
        roast_date: Roast date as string
        bag_size: Bag size (e.g. "250g")
        roaster_notes: Tasting notes from the roaster
        freshness_note: Freshness window note
        approach: Narrative brewing approach — profile choice, pressure logic,
            starting parameters, and reasoning (for 'create')
        entry_date: Date of journal entry (for 'log_entry', defaults to today)
        entry_headline: Journal entry headline, e.g. "Grind 10, Lever Decline [AI] — 4/5"
            (for 'log_entry')
        entry_body: Freeform analysis — what worked, what didn't, what to try next
            (for 'log_entry')
        content: Full markdown content (for 'update' action)

    Returns:
        JSON string with result
    """
    logger.info("manage_coffee_called", action=action, coffee_name=coffee_name)

    def _find_coffee_file(name: str) -> str | None:
        """Find an existing coffee file by name.

        Tries exact match, sanitized name, and partial match against
        available files. This handles the case where create uses
        'roaster + coffee_name' but user passes just 'coffee_name'.

        Returns:
            Filename (without .md) if found, None otherwise
        """
        # Exact match (user passed the filename directly)
        if coffee_storage.exists(name):
            return name

        # Try sanitized version
        sanitized = _sanitize_filename(name)
        if coffee_storage.exists(sanitized):
            return sanitized

        # Search for partial match (e.g. "azul" matches "jack-lefleur-azul")
        available = coffee_storage.list_files()
        sanitized_lower = sanitized.lower()
        for f in available:
            if sanitized_lower in f.lower():
                return f

        return None

    try:
        if action == "list":
            files = coffee_storage.list_files()
            return json.dumps({
                "success": True,
                "action": "list",
                "coffees": files,
                "count": len(files),
                "resource_uri": "gaggimate://coffees",
            })

        elif action == "read":
            if not coffee_name:
                return json.dumps({
                    "success": False,
                    "error": "coffee_name is required for 'read'"
                })
            filename = _find_coffee_file(coffee_name)
            if filename is None:
                available = coffee_storage.list_files()
                return json.dumps({
                    "success": False,
                    "error": f"Coffee '{coffee_name}' not found. Available: {', '.join(available)}"
                })
            content_text = coffee_storage.read(filename)
            return json.dumps({
                "success": True,
                "action": "read",
                "coffee_name": filename,
                "content": content_text,
            })

        elif action == "create":
            if not coffee_name or not roaster:
                return json.dumps({
                    "success": False,
                    "error": "coffee_name and roaster are required for 'create'"
                })

            filename = _sanitize_filename(f"{roaster} {coffee_name}")
            if coffee_storage.exists(filename):
                return json.dumps({
                    "success": False,
                    "error": f"Coffee file '{filename}.md' already exists. Use 'update' or 'log_shot' instead."
                })

            md_content = build_coffee_markdown(
                coffee_name=coffee_name,
                roaster=roaster,
                origin=origin or "",
                process=process or "",
                variety=variety or "",
                roast_level=roast_level or "",
                roast_date=roast_date or "",
                bag_size=bag_size or "",
                roaster_notes=roaster_notes or "",
                freshness_note=freshness_note or "",
                approach=approach or "",
            )

            path = coffee_storage.write(filename, md_content)
            return json.dumps({
                "success": True,
                "action": "create",
                "filename": f"{filename}.md",
                "path": str(path),
                "resource_uri": f"gaggimate://coffees/{filename}.md",
            })

        elif action == "log_entry":
            if not coffee_name:
                return json.dumps({
                    "success": False,
                    "error": "coffee_name is required to identify which coffee file to update"
                })

            filename = _find_coffee_file(coffee_name)
            if filename is None:
                return json.dumps({
                    "success": False,
                    "error": f"Coffee file not found for '{coffee_name}'. "
                             f"Available: {', '.join(coffee_storage.list_files())}"
                })

            existing = coffee_storage.read(filename)

            updated = append_journal_entry(
                existing,
                date=entry_date or "",
                headline=entry_headline or "",
                body=entry_body or "",
            )

            coffee_storage.write(filename, updated)
            return json.dumps({
                "success": True,
                "action": "log_entry",
                "filename": f"{filename}.md",
                "resource_uri": f"gaggimate://coffees/{filename}.md",
            })

        elif action == "update":
            if not coffee_name:
                return json.dumps({
                    "success": False,
                    "error": "coffee_name is required for 'update'"
                })
            if not content:
                return json.dumps({
                    "success": False,
                    "error": "content is required for 'update'. Provide the full markdown."
                })

            filename = _find_coffee_file(coffee_name)
            if filename is None:
                # For update, allow creating with sanitized name
                filename = _sanitize_filename(coffee_name)

            path = coffee_storage.write(filename, content)
            return json.dumps({
                "success": True,
                "action": "update",
                "filename": f"{filename}.md",
                "path": str(path),
                "resource_uri": f"gaggimate://coffees/{filename}.md",
            })

        elif action == "delete":
            if not coffee_name:
                return json.dumps({
                    "success": False,
                    "error": "coffee_name is required for 'delete'"
                })

            filename = _find_coffee_file(coffee_name)
            if filename is None:
                return json.dumps({
                    "success": False,
                    "error": f"Coffee file not found for '{coffee_name}'"
                })

            deleted = coffee_storage.delete(filename)

            if deleted:
                return json.dumps({
                    "success": True,
                    "action": "delete",
                    "message": f"Deleted coffee file for '{coffee_name}'"
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Coffee file not found for '{coffee_name}'"
                })

        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown action '{action}'. Use: list, read, create, log_entry, update, delete"
            })

    except Exception as e:
        error_msg = str(e) or f"{type(e).__name__} (no message)"
        logger.error("manage_coffee_error", action=action, error=error_msg, exception_type=type(e).__name__)
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {error_msg}",
            "exception_type": type(e).__name__
        })


@mcp.tool()
async def manage_user_setup(
    action: str = "read",
    content: Optional[str] = None,
) -> str:
    """Manage the user setup file (equipment, preferences, workflow).

    The user setup file stores equipment details, flavor preferences, and workflow
    notes. Read it via gaggimate://user/setup resource; write/update via this tool.

    Args:
        action: Action to perform:
            - 'read': Read current user setup (returns content or template guidance)
            - 'write': Create or overwrite the user setup file
        content: Full markdown content for the user setup file (required for 'write')

    Returns:
        JSON string with result
    """
    logger.info("manage_user_setup_called", action=action)

    try:
        if action == "read":
            existing = user_storage.read("user-setup")
            if existing is not None:
                return json.dumps({
                    "success": True,
                    "action": "read",
                    "content": existing,
                    "resource_uri": "gaggimate://user/setup",
                })
            else:
                # Check for example template
                example = user_storage.read("user-setup.example")
                return json.dumps({
                    "success": True,
                    "action": "read",
                    "content": None,
                    "message": "No user setup file found. Use action='write' to create one.",
                    "example_available": example is not None,
                    "resource_uri": "gaggimate://user/setup",
                })

        elif action == "write":
            if not content:
                return json.dumps({
                    "success": False,
                    "error": "content is required for 'write'. Provide full markdown for user-setup.md."
                })

            path = user_storage.write("user-setup", content)
            return json.dumps({
                "success": True,
                "action": "write",
                "path": str(path),
                "resource_uri": "gaggimate://user/setup",
                "message": "User setup saved successfully."
            })

        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown action '{action}'. Use: read, write"
            })

    except Exception as e:
        error_msg = str(e) or f"{type(e).__name__} (no message)"
        logger.error("manage_user_setup_error", action=action, error=error_msg, exception_type=type(e).__name__)
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {error_msg}",
            "exception_type": type(e).__name__
        })


@mcp.tool()
async def manage_grind_map(
    action: str = "read",
    content: Optional[str] = None,
    coffee: Optional[str] = None,
    roast: Optional[str] = None,
    process: Optional[str] = None,
    origin: Optional[str] = None,
    days_off_roast: Optional[str] = None,
    grind: Optional[str] = None,
    profile: Optional[str] = None,
    ratio: Optional[str] = None,
    temp: Optional[str] = None,
    rating: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """Manage the grind map — a record of successful grind settings.

    The grind map tracks which grind settings worked for which coffees,
    building a personal reference over time. Read via gaggimate://user/grind-map
    resource; write/update via this tool.

    Args:
        action: Action to perform:
            - 'read': Read current grind map
            - 'write': Create or overwrite the entire grind map
            - 'add_entry': Append a successful grind setting row
            - 'init': Initialize a fresh grind map from template
        content: Full markdown content (for 'write' action)
        coffee: Coffee name (for 'add_entry')
        roast: Roast level (for 'add_entry')
        process: Processing method (for 'add_entry')
        origin: Origin country (for 'add_entry')
        days_off_roast: Days since roast (for 'add_entry', default "—")
        grind: Grind setting (for 'add_entry')
        profile: Profile used (for 'add_entry')
        ratio: Brew ratio (for 'add_entry', e.g. "1:2")
        temp: Temperature (for 'add_entry', e.g. "93°C")
        rating: Rating (for 'add_entry', e.g. "4")
        date: Date (for 'add_entry', defaults to today)

    Returns:
        JSON string with result
    """
    logger.info("manage_grind_map_called", action=action)

    try:
        if action == "read":
            existing = user_storage.read("grind-map")
            if existing is not None:
                return json.dumps({
                    "success": True,
                    "action": "read",
                    "content": existing,
                    "resource_uri": "gaggimate://user/grind-map",
                })
            else:
                return json.dumps({
                    "success": True,
                    "action": "read",
                    "content": None,
                    "message": "No grind map found. Use action='init' to create one.",
                    "resource_uri": "gaggimate://user/grind-map",
                })

        elif action == "init":
            if user_storage.exists("grind-map"):
                return json.dumps({
                    "success": False,
                    "error": "Grind map already exists. Use 'add_entry' to add rows, or 'write' to overwrite."
                })

            md = build_grind_map_markdown()
            path = user_storage.write("grind-map", md)
            return json.dumps({
                "success": True,
                "action": "init",
                "path": str(path),
                "resource_uri": "gaggimate://user/grind-map",
                "message": "Grind map initialized."
            })

        elif action == "add_entry":
            existing = user_storage.read("grind-map")
            if not existing:
                # Auto-initialize if missing
                existing = build_grind_map_markdown()

            updated = append_grind_entry(
                existing,
                coffee=coffee or "",
                roast=roast or "",
                process=process or "",
                origin=origin or "",
                days_off_roast=days_off_roast or "—",
                grind=grind or "",
                profile=profile or "",
                ratio=ratio or "",
                temp=temp or "",
                rating=rating or "",
                date=date or "",
            )

            path = user_storage.write("grind-map", updated)
            return json.dumps({
                "success": True,
                "action": "add_entry",
                "path": str(path),
                "resource_uri": "gaggimate://user/grind-map",
                "message": "Grind entry added."
            })

        elif action == "write":
            if not content:
                return json.dumps({
                    "success": False,
                    "error": "content is required for 'write'. Provide full markdown."
                })

            path = user_storage.write("grind-map", content)
            return json.dumps({
                "success": True,
                "action": "write",
                "path": str(path),
                "resource_uri": "gaggimate://user/grind-map",
                "message": "Grind map saved."
            })

        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown action '{action}'. Use: read, write, add_entry, init"
            })

    except Exception as e:
        error_msg = str(e) or f"{type(e).__name__} (no message)"
        logger.error("manage_grind_map_error", action=action, error=error_msg, exception_type=type(e).__name__)
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {error_msg}",
            "exception_type": type(e).__name__
        })


@mcp.tool()
async def manage_brewing_insights(
    action: str = "read",
    content: Optional[str] = None,
) -> str:
    """Manage the brewing insights file — cross-coffee patterns and learnings.

    The brewing insights file captures what you've learned across coffees:
    which origin/process/roast combinations respond to which profiles, general
    patterns, and links to specific coffee files for details. Read it via
    gaggimate://user/brewing-insights resource; write/update via this tool.

    The agent should update this file when meaningful patterns emerge from
    brewing sessions — not after every single shot, but when there's a
    genuine learning to record (e.g. "Brazilian naturals benefit from
    declining pressure profiles").

    Args:
        action: Action to perform:
            - 'read': Read current brewing insights
            - 'write': Create or overwrite the entire brewing insights file
            - 'init': Initialize a fresh brewing insights file from template

    Returns:
        JSON string with result
    """
    logger.info("manage_brewing_insights_called", action=action)

    try:
        if action == "read":
            existing = user_storage.read("brewing-insights")
            if existing is not None:
                return json.dumps({
                    "success": True,
                    "action": "read",
                    "content": existing,
                    "resource_uri": "gaggimate://user/brewing-insights",
                })
            else:
                return json.dumps({
                    "success": True,
                    "action": "read",
                    "content": None,
                    "message": "No brewing insights file found. Use action='init' to create one.",
                    "resource_uri": "gaggimate://user/brewing-insights",
                })

        elif action == "init":
            if user_storage.exists("brewing-insights"):
                return json.dumps({
                    "success": False,
                    "error": "Brewing insights file already exists. Use 'write' to overwrite."
                })

            md = build_brewing_insights_markdown()
            path = user_storage.write("brewing-insights", md)
            return json.dumps({
                "success": True,
                "action": "init",
                "path": str(path),
                "resource_uri": "gaggimate://user/brewing-insights",
                "message": "Brewing insights file initialized."
            })

        elif action == "write":
            if not content:
                return json.dumps({
                    "success": False,
                    "error": "content is required for 'write'. Provide full markdown."
                })

            path = user_storage.write("brewing-insights", content)
            return json.dumps({
                "success": True,
                "action": "write",
                "path": str(path),
                "resource_uri": "gaggimate://user/brewing-insights",
                "message": "Brewing insights saved."
            })

        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown action '{action}'. Use: read, write, init"
            })

    except Exception as e:
        error_msg = str(e) or f"{type(e).__name__} (no message)"
        logger.error("manage_brewing_insights_error", action=action, error=error_msg, exception_type=type(e).__name__)
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {error_msg}",
            "exception_type": type(e).__name__
        })


@mcp.tool()
async def read_knowledge(
    action: str,
    filename: Optional[str] = None,
) -> str:
    """Access espresso knowledge files.

    Knowledge files contain authoritative espresso guidance — temperature tables,
    pressure recommendations, tasting diagnosis, profile templates, and more.
    Always prefer these over general training data.

    Args:
        action: "list" to list available files, "read" to read a specific file.
        filename: Name of the knowledge file to read (with or without .md extension).
            Supports subdirectory files like "profiles/EXAMPLES" or "diagnostics/DIAGNOSTIC_TREES".
            Required when action is "read".

    Returns:
        JSON string with file listing or file content.

    Examples:
        - read_knowledge(action="list") → list all knowledge files
        - read_knowledge(action="read", filename="ESPRESSO_BREWING_BASICS") → read brewing basics
        - read_knowledge(action="read", filename="PRESSURE_GUIDE") → read pressure guide
        - read_knowledge(action="read", filename="profiles/EXAMPLES") → read profile examples
        - read_knowledge(action="read", filename="diagnostics/TELEMETRY_PATTERNS") → read telemetry patterns
    """
    knowledge_dir = Path(config.knowledge_dir)
    logger.info("read_knowledge_called", action=action, filename=filename)

    try:
        if action == "list":
            files = _list_markdown_files(knowledge_dir)
            if not files:
                return json.dumps({
                    "success": True,
                    "action": "list",
                    "files": [],
                    "message": "No knowledge files found.",
                })

            return json.dumps({
                "success": True,
                "action": "list",
                "files": [f["filename"] for f in files],
                "message": f"Found {len(files)} knowledge files. Call read_knowledge(action='read', filename='...') to read one.",
            })

        if action == "read":
            if not filename:
                return json.dumps({
                    "success": False,
                    "error": "filename is required when action is 'read'.",
                })

            content = _read_markdown_file(knowledge_dir, filename)
            return json.dumps({
                "success": True,
                "action": "read",
                "filename": filename,
                "content": content,
            })

        return json.dumps({
            "success": False,
            "error": f"Unknown action '{action}'. Use 'list' or 'read'.",
        })

    except FileNotFoundError as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })
    except Exception as e:
        error_msg = str(e) or f"{type(e).__name__} (no message)"
        logger.error("read_knowledge_error", action=action, filename=filename, error=error_msg, exception_type=type(e).__name__)
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {error_msg}",
            "exception_type": type(e).__name__
        })

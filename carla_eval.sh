#!/bin/bash
export CARLA_ROOT=PATH_TO_CARLA
export DTCP_PATH=PATH_TO_DTCP_FILES
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${DTCP_PATH}/leaderboard
export PYTHONPATH=$PYTHONPATH:${DTCP_PATH}/leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:${DTCP_PATH}/scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000
export TM_PORT=8000
export DEBUG_CHALLENGE=0
export REPETITIONS=3 # multiple runs
export RESUME=True


# DTCP evaluation
export ROUTES=leaderboard/data/evaluation_routes/routes_lav_valid.xml
export TEAM_AGENT=leaderboard/team_code/dtcp_agent.py
export TEAM_CONFIG= PATH_TO_MODEL_CKPT
export CHECKPOINT_ENDPOINT=results_DTCP.json
export SCENARIOS=leaderboard/data/scenarios/all_towns_traffic_scenarios.json
export SAVE_PATH=PATH_TO_SAVE_RESULTS
export RECORD_PATH=PATH_TO_SAVE_LOGS # simulation log file



python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--trafficManagerSeed=${TM_SEED} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME}

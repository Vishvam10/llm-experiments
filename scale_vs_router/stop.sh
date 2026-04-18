#!/usr/bin/env bash

echo "Stopping MLX servers ..."

kill $(cat big.pid)
kill $(cat code.pid)
kill $(cat math.pid)
kill $(cat logic.pid)

rm *.pid

echo "Stopped."
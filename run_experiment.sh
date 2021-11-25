#!/bin/bash

for ((i=0; i<10; i++))
do
  python main.py \
  --env "HalfCheetah-v3" \
  --max_timesteps 5000000 \
  --seed $i
done

for ((i=0; i<10; i++))
do
  python main.py \
  --env "Walker2d-v3" \
  --max_timesteps 5000000 \
  --seed $i
done

for ((i=0; i<10; i++))
do
  python main.py \
  --env "Ant-v3" \
  --max_timesteps 5000000 \
  --seed $i
done

for ((i=0; i<10; i++))
do
  python main.py \
  --env "Humanoid-v3" \
  --max_timesteps 10000000 \
  --seed $i
done

for ((i=0; i<10; i++))
do
  python main.py \
  --env "Hopper-v3" \
  --max_timesteps 3000000 \
  --seed $i
done

for ((i=0; i<10; i++))
do
  python main.py \
  --env "Swimmer-v3" \
  --max_timesteps 3000000 \
  --seed $i
done



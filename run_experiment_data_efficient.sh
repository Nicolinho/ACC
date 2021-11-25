#!/bin/bash

for ((i=0; i<10; i++))
do
  python main.py \
  --env "HalfCheetah-v3" \
  --max_timesteps 1000000 \
  --num_critic_updates 4 \
  --seed $i
done


for ((i=0; i<10; i++))
do
  python main.py \
  --env "Walker2d-v3" \
  --max_timesteps 1000000 \
  --num_critic_updates 4 \
  --seed $i
done

for ((i=0; i<10; i++))
do
  python main.py \
  --env "Ant-v3" \
  --max_timesteps 1000000 \
  --num_critic_updates 4 \
  --seed $i
done

for ((i=0; i<10; i++))
do
  python main.py \
  --env "Humanoid-v3" \
  --max_timesteps 1000000 \
  --num_critic_updates 4 \
  --seed $i
done

for ((i=0; i<10; i++))
do
  python main.py \
  --env "Hopper-v3" \
  --max_timesteps 500000 \
  --num_critic_updates 4 \
  --seed $i
done

for ((i=0; i<10; i++))
do
  python main.py \
  --env "Swimmer-v3" \
  --max_timesteps 500000 \
  --num_critic_updates 4 \
  --seed $i
done



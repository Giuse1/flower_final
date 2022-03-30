#!/bin/bash
for i in {0..4}
do
  python3 client.py -i $i &
done
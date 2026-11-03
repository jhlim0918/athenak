#!/bin/bash
# usage: rate.sh LOG -> asymptotic contraction (geometric mean over iterations 40..59) and final defect
grep "MG iteration" $1 | awk '{print $NF}' | awk 'NR==41{a=$1} NR==60{b=$1} END{printf "rate(40-59)=%.4f  def60=%.3e\n", (b/a)^(1/19), b}'

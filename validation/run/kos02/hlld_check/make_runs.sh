#!/bin/bash
# Isothermal HLLD noise check at KOS02's beta = cs^2/vA^2 = 1 (vA = cs, plasma beta 2), with the
# October controls vA = 1.5 (plasma beta 0.89, unstable then) and vA = 0.5 (beta 8, stable).
# Uniform rho = 1, cs = 1, no gravity, 1e-8 white velocity noise, 128 x 8 x 8 cells in 1 x 1/8 x 1/8.
# B along x2 (b_angle = 90: perpendicular to x1, as KOS02's azimuthal field) or along x1 (b_angle = 0).
ATH=$HOME/athenak-worktrees/fargo-amr/build-fft/src/athena
SRC=$HOME/athenak-worktrees/fargo-amr/inputs/tests/magneto_jeans.athinput
for rs in hlld hlle; do for rec in plm ppm4; do for b0 in 0.5 1.0 1.5; do for ang in 90 0; do
  [ $rs = hlle ] && [ $rec = ppm4 ] && continue
  n=${rs}_${rec}_b${b0}_a${ang}; mkdir -p $n; cd $n
  sed -e 's/^eos .*/eos = isothermal\niso_sound_speed = 1.0/' -e '/^gamma /d' \
      -e "s/^reconstruct .*/reconstruct = $rec/" -e "s/^rsolver .*/rsolver = $rs/" \
      -e 's/^nx1       = 64/nx1       = 128/' -e 's/^integrator = rk3/integrator = rk2/' \
      -e 's/^self_gravity = true/self_gravity = false/' -e 's/^tlim       = 1.0/tlim       = 3.0/' \
      -e 's/^amp       = 1.0e-4.*/amp       = 0.0\nnoise_amp = 1.0e-8/' -e "s/^b0        = 1.0.*/b0        = $b0/" \
      -e "s/^b_angle   = 90.0.*/b_angle   = $ang/" -e '/^p0 /d' -e 's/^user_hist = true/user_hist = false/' $SRC > in.athinput
  $ATH -i in.athinput > log 2>&1; echo "$n exit=$?"; cd ..
done; done; done; done

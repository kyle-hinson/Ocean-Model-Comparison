
* clone E3SM from de's branch: 
  https://github.com/dengwirda/E3SM/tree/dengwirda/wall-slip
- since we need the no slip walls...



* when updating namelist:
- config_dt should be same mins as resolution km
- config_btr_dt = config_dt / 4  (should scale linearly with resolution)
- config_mom_del2 = 33.3 * (resolution [m] / 30000) ^ 1  (round to non-crazy number...)
- config_mom_del4 = 3.33E11 * (resolution [m] / 30000) ^ 3



mkdir 10km
python3 channel_case.py \
--case-name=10km/channel \
--num-xcell=10 \
--num-ycell=34 \
--len-edges=10000. \
--wind-stress=0.100

cd 10km
cp ocean_model, namelist and streams to .
adjust namelist time-steps, viscosity and pio settings
srun -n 1 ./ocean_model -n namelist.ocean -s streams.ocean



mkdir 5km
python3 channel_case.py \
--case-name=5km/channel \
--num-xcell=20 \
--num-ycell=70 \
--len-edges=5000. \
--wind-stress=0.100

cd 5km
gpmetis channel_graph.info 4
cp ocean_model, namelist and streams to .
adjust namelist time-steps, viscosity and pio settings
srun -n 4 ./ocean_model -n namelist.ocean -s streams.ocean
    


mkdir 2km
python3 channel_case.py \
--case-name=2km/channel \
--num-xcell=50 \
--num-ycell=174 \
--len-edges=2000. \
--wind-stress=0.100

cd 2km
gpmetis channel_graph.info 40
cp ocean_model, namelist and streams to .
adjust namelist time-steps, viscosity and pio settings
srun -n 40 ./ocean_model -n namelist.ocean -s streams.ocean



mkdir 1km
python3 channel_case.py \
--case-name=1km/channel \
--num-xcell=100 \
--num-ycell=346 \
--len-edges=1000. \
--wind-stress=0.100

cd 1km
gpmetis channel_graph.info 128
cp ocean_model, namelist and streams to 1km
adjust namelist time-steps, viscosity and pio settings
srun -n 128 ./ocean_model -n namelist.ocean -s streams.ocean


    
mkdir 500m
python3 channel_case.py \
--case-name=500m/channel \
--num-xcell=200 \
--num-ycell=692 \
--len-edges=500. \
--wind-stress=0.100

cd 500m
gpmetis channel_graph.info 512
cp ocean_model, namelist and streams to .
adjust namelist time-steps, viscosity and pio settings
srun -n 512 ./ocean_model -n namelist.ocean -s streams.ocean



mkdir 200m
python3 channel_case.py \
--case-name=200m/channel \
--num-xcell=500 \
--num-ycell=1732 \
--len-edges=200. \
--wind-stress=0.100

cd 200m
gpmetis channel_graph.info 2048
cp ocean_model, namelist and streams to .
adjust namelist time-steps, viscosity and pio settings
srun -n 2048 ./ocean_model -n namelist.ocean -s streams.ocean


cd 100 m
python3 channel_case.py \
--case-name=channel \
--num-xcell=1000 \
--num-ycell=3462 \
--len-edges=100. \
--wind-stress=0.100

cd 100m
gpmetis channel_graph.info 2048
cp ocean_model, namelist and streams to .
adjust namelist time-steps, viscosity and pio settings
srun -n 2048 ./ocean_model -n namelist.ocean -s streams.ocean

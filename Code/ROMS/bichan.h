/*
** Options for the Channel Instability case used for ROMS/MPAS-O comparisons.
**
** Application flag:   CHANNEL
** Input script:       channel.in
*/

#define ROMS_MODEL 

#define UV_ADV
#define UV_COR
#define UV_QDRAG
#define UV_VIS2
#define MIX_S_UV
#define SPLINES_VDIFF
#define SPLINES_VVISC
#define DJ_GRADPS
#define TS_DIF2
#define MIX_S_TS

#define SALINITY
#define SOLVE3D

#undef AVERAGES
#undef DIAGNOSTICS_TS
#undef DIAGNOSTICS_UV

#define ANA_GRID
#define ANA_INITIAL
#define ANA_SMFLUX
#define ANA_STFLUX
#define ANA_SSFLUX
#define ANA_BTFLUX
#define ANA_BSFLUX

#define GLS_MIXING
#define KANTHA_CLAYSON
#define N2S2_HORAVG
#define RI_SPLINES

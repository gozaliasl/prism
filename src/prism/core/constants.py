"""
Physical and instrumental constants shared across the PRISM pipeline.

All values in SI unless otherwise noted.  Import from here rather than
sprinkling magic numbers through individual modules.
"""

# Speed of light
C_LIGHT_KM_S: float = 299792.458          # km s⁻¹  (exact, IAU 2012)
C_LIGHT_M_S:  float = 299792458.0         # m s⁻¹
C_LIGHT_MPC_PER_DAY: float = C_LIGHT_KM_S * 86400.0 / 3.085677581e22  # Mpc day⁻¹

# Gravitational constant
G_SI: float = 6.67430e-11  # m³ kg⁻¹ s⁻²

# Unit conversions
KM_PER_MPC:   float = 3.085677581e19
ARCSEC_TO_RAD: float = 4.84813681e-6

# Default cosmology (flat ΛCDM, consistent with planck2020a reference)
H0_DEFAULT:   float = 70.0   # km s⁻¹ Mpc⁻¹
OMEGA_M:      float = 0.3
OMEGA_LAMBDA: float = 0.7

# Canonical JWST NIRCam SW pixel scale.
# *Telescope-specific values live in telescope_configs inside default_config.yaml.*
# This constant is provided as a cross-check reference only.
JWST_NIRCAM_SW_PIXEL_SCALE: float = 0.031  # arcsec pixel⁻¹

# =============================================================================
# FINAL INTEGRATED ANALYSIS — COMPLETE & SELF‑CONTAINED (CORRECTED)
# Pantheon+ Directional Hubble Analysis
# =============================================================================
import csv, json, math, time
import numpy as np

try:
    from scipy.stats import chi2 as scipy_chi2
except ImportError:
    scipy_chi2 = None

# ========================= CONFIGURATION =========================
CATALOG_FILE = "Pantheon+SH0ES.dat"
COV_FILE     = "Pantheon+SH0ES_STAT+SYS.cov.txt"

OUTPUT_JSON     = "anisotropy_results_final.json"
OUTPUT_CSV      = "anisotropy_alignment_final.csv"
OUTPUT_INJ_JSON = "injection_test_results_final.json"
OUTPUT_RMAT     = "response_matrix_R_dip_z003.csv"
OUTPUT_SCATTER  = "injection_scatter_data.csv"
OUTPUT_TOMO     = "cumulative_tomography_results.csv"
OUTPUT_BULK_TOMO = "bulkflow_tomography.csv"
OUTPUT_SKY_MAP   = "profile_sky_map.csv"
OUTPUT_SURVEY    = "survey_split_summary.csv"
OUTPUT_SHAPLEY   = "shapley_cone_test.csv"

C_LIGHT = 299792.458
H0 = 73.2
OMEGA_M = 0.30

NSIM_NULL = 100000
NSIM_AXIS = 100000
NSIM_INJ  = 200
N_Z_BINS  = 4
NSIM_HEMI = 5000
JACK_FRAC_REMOVE = 0.10
CMB_DIPOLE_RA = 168.0
CMB_DIPOLE_DEC = -7.0
NSIM_TOMO_NULL = 2000

SHAPLEY_RA = 172.0
SHAPLEY_DEC = -54.0

CHOLESKY_JITTERS = [0.0, 1e-12, 1e-10, 1e-8, 1e-6]

KNOWN_STRUCTURES = {
    "Shapley Supercluster": (172.0, -54.0),
    "Horologium-Reticulum": (77.5, -54.0),
    "Local Void": (270.0, 30.0),
    "2M++ Galaxy Dipole": (168.0, -7.0),
    "CMB Dipole": (168.0, -7.0),
}

# ========================= UTILITIES =========================
def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def chi2_sf(x, k):
    if x <= 0 or k <= 0: return 1.0
    if scipy_chi2 is not None: return float(scipy_chi2.sf(x, k))
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return max(0.0, min(1.0, 1.0 - norm_cdf(z)))

def aic(chi2, k): return chi2 + 2.0 * k
def bic(chi2, k, n): return chi2 + k * math.log(n)

def empirical_p(null_vals, observed):
    null_vals = np.asarray(null_vals)
    return (1.0 + np.sum(null_vals >= observed)) / (len(null_vals) + 1.0)

def angular_distance(v1, v2):
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return None
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(c))))

def vec_to_radec(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0: return None, None
    x, y, z = v / n
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return float(ra), float(dec)

def axis_reliability(pval, r95):
    if pval is None or r95 is None: return "unknown"
    if pval > 0.05 or r95 > 90.0: return "unreliable"
    if pval <= 0.01 and r95 <= 30.0: return "strong"
    return "moderate"

def trapz_compat(y, x):
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)

def make_json_serializable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, dict): return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple): return tuple(make_json_serializable(v) for v in obj)
    return obj

# ========================= DATA LOADING =========================
def load_catalog(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    header_idx = next(i for i, ln in enumerate(lines) if "z" in ln.lower() and "ra" in ln.lower())
    cols = lines[header_idx].split()
    colmap = {c: j for j, c in enumerate(cols)}
    for key in ["zHD","RA","DEC","m_b_corr","IS_CALIBRATOR"]:
        if key not in colmap:
            for c in cols:
                if c.lower() == key.lower():
                    colmap[key] = colmap[c]
                    break
            else: raise ValueError(f"Column {key} missing")
    survey_col = None
    for c in cols:
        if "SURVEY" in c.upper():
            survey_col = colmap[c]
            break
    data = []
    for ln in lines[header_idx+1:]:
        sp = ln.split()
        if len(sp) < len(cols): continue
        try:
            entry = {"z": float(sp[colmap["zHD"]]),
                     "ra": float(sp[colmap["RA"]]),
                     "dec": float(sp[colmap["DEC"]]),
                     "mu": float(sp[colmap["m_b_corr"]]),
                     "is_ceph": (float(sp[colmap["IS_CALIBRATOR"]]) == 1.0)}
            if survey_col is not None:
                entry["survey"] = sp[survey_col]
            data.append(entry)
        except Exception: continue
    return data

def load_covariance(filename, n):
    vals = []
    with open(filename, "r", encoding="utf-8") as f:
        for ln in f: vals.extend(float(tok) for tok in ln.split())
    if len(vals) == n*n + 1: vals = vals[1:]
    return np.array(vals, dtype=float).reshape((n, n))

# ========================= COSMOLOGY & GEOMETRY =========================
def e_z_flat_lcdm(z, omega_m=OMEGA_M):
    return np.sqrt(omega_m * (1.0+z)**3 + (1.0 - omega_m))

def comoving_distance_flat_lcdm(z, H0=H0, omega_m=OMEGA_M):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    for i, zi in enumerate(z):
        if zi <= 0.0: out[i] = 0.0; continue
        zz = np.linspace(0.0, zi, max(256, int(512*zi))+1)
        out[i] = (C_LIGHT / H0) * trapz_compat(1.0 / e_z_flat_lcdm(zz, omega_m), zz)
    return out

def mu_iso_lcdm(z, H0=H0, omega_m=OMEGA_M):
    dl_mpc = np.clip((1.0 + np.asarray(z)) * comoving_distance_flat_lcdm(z, H0, omega_m), 1e-12, None)
    return 5.0 * np.log10(dl_mpc) + 25.0

def sky_unit_vectors(ra_deg, dec_deg):
    ra, dec = np.radians(np.asarray(ra_deg)), np.radians(np.asarray(dec_deg))
    return np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

def quadrupole_basis_traceless(nvec):
    nx, ny, nz = nvec[:,0], nvec[:,1], nvec[:,2]
    return np.column_stack([nx*nx - ny*ny, 2.0*nz*nz - nx*nx - ny*ny,
                            2.0*nx*ny, 2.0*nx*nz, 2.0*ny*nz])

# ========================= GLS CORE =========================
def prepare_whitening(C):
    I = np.eye(C.shape[0])
    for eps in CHOLESKY_JITTERS:
        try: return np.linalg.cholesky(C if eps == 0.0 else (C + eps * I)), eps
        except np.linalg.LinAlgError: continue
    raise np.linalg.LinAlgError("Cholesky failed.")

def build_model_cache(M, L):
    Mw = np.linalg.solve(L, M)
    U, s, Vt = np.linalg.svd(Mw, full_matrices=False)
    rank = int(np.sum(s > np.finfo(float).eps * max(Mw.shape) * s[0]))
    s_r, Vt_r, U_r = s[:rank], Vt[:rank, :], U[:, :rank]
    Mw_pinv = (Vt_r.T / s_r) @ U_r.T
    return {"M": M, "Mw": Mw, "Mw_pinv": Mw_pinv, "rank": rank,
            "cond": float(s[0]/s_r[-1]), "singular_values": s}

def fit_from_cache_yw(yw, cache):
    beta = cache["Mw_pinv"] @ yw
    chi2 = float(np.sum((yw - cache["Mw"] @ beta)**2))
    return beta, chi2

# ========================= DESIGNS & AXIS =========================
def build_joint_designs(mu_iso, nvec):
    base = np.column_stack([np.ones_like(mu_iso), mu_iso])
    qbasis = quadrupole_basis_traceless(nvec)
    return {"monopole": base,
            "dipole": np.column_stack([base, nvec]),
            "quadrupole": np.column_stack([base, qbasis]),
            "dipole+quadrupole": np.column_stack([base, nvec, qbasis])}

def dipole_axis_from_beta(beta, start=2):
    v = beta[start:start+3]; amp = float(np.linalg.norm(v))
    ra, dec = vec_to_radec(v)
    return {"vec": v, "amp": amp, "ra_deg": ra, "dec_deg": dec}

def quadrupole_axis_from_beta(beta, start=2):
    q1, q2, q3, q4, q5 = beta[start:start+5]
    Q = np.array([[q1 - q2, q3, q4],
                  [q3, -q1 - q2, q5],
                  [q4, q5, 2.0*q2]])
    vals, vecs = np.linalg.eigh(Q)
    idx = int(np.argmax(np.abs(vals)))
    v = vecs[:, idx]
    ra, dec = vec_to_radec(v)
    return {"vec": v, "eigval": float(vals[idx]), "ra_deg": ra, "dec_deg": dec}

def axis_uncertainty_bootstrap(cache, beta_hat, start, is_quad=False, nsim=NSIM_AXIS, rng=None):
    if rng is None: rng = np.random.default_rng()
    if is_quad:
        ax0 = quadrupole_axis_from_beta(beta_hat, start)
        if ax0 is None: return None
        ref = ax0["vec"] / np.linalg.norm(ax0["vec"])
    else:
        ax0 = dipole_axis_from_beta(beta_hat, start)
        if ax0 is None or ax0["amp"] <= 0: return None
        ref = ax0["vec"] / np.linalg.norm(ax0["vec"])
    mean_yw = cache["Mw"] @ beta_hat
    n = len(mean_yw)
    angs = np.empty(nsim); vals = np.empty(nsim); m = 0
    for _ in range(nsim):
        yw_mock = mean_yw + rng.standard_normal(n)
        beta, _ = fit_from_cache_yw(yw_mock, cache)
        if is_quad:
            ax = quadrupole_axis_from_beta(beta, start)
            if ax is None: continue
            v = ax["vec"] / np.linalg.norm(ax["vec"])
            a = angular_distance(ref, v)
            if a is not None: angs[m] = a; vals[m] = ax["eigval"]; m += 1
        else:
            ax = dipole_axis_from_beta(beta, start)
            if ax is None or ax["amp"] <= 0: continue
            v = ax["vec"] / np.linalg.norm(ax["vec"])
            a = angular_distance(ref, v)
            if a is not None: angs[m] = a; vals[m] = ax["amp"]; m += 1
    if m == 0: return None
    angs, vals = angs[:m], vals[:m]
    if is_quad:
        return {"r68_deg": float(np.percentile(angs, 68)), "r95_deg": float(np.percentile(angs, 95)),
                "eig_p50": float(np.percentile(vals, 50)), "eig_p16": float(np.percentile(vals, 16)),
                "eig_p84": float(np.percentile(vals, 84))}
    else:
        return {"r68_deg": float(np.percentile(angs, 68)), "r95_deg": float(np.percentile(angs, 95)),
                "amp_p50": float(np.percentile(vals, 50)), "amp_p16": float(np.percentile(vals, 16)),
                "amp_p84": float(np.percentile(vals, 84))}

def joint_axis_uncertainty(cache, beta_hat, dstart, qstart, nsim=NSIM_AXIS, rng=None):
    if rng is None: rng = np.random.default_rng()
    axd0 = dipole_axis_from_beta(beta_hat, dstart)
    axq0 = quadrupole_axis_from_beta(beta_hat, qstart)
    mean_yw = cache["Mw"] @ beta_hat; n = len(mean_yw)
    md = mq = 0
    angs_d = np.empty(nsim); amps_d = np.empty(nsim)
    angs_q = np.empty(nsim); eigs_q = np.empty(nsim)
    for _ in range(nsim):
        yw_mock = mean_yw + rng.standard_normal(n)
        beta, _ = fit_from_cache_yw(yw_mock, cache)
        if axd0 and axd0["amp"] > 0:
            axd = dipole_axis_from_beta(beta, dstart)
            if axd and axd["amp"] > 0:
                vd = axd["vec"] / np.linalg.norm(axd["vec"])
                ang = angular_distance(axd0["vec"] / np.linalg.norm(axd0["vec"]), vd)
                if ang is not None: angs_d[md] = ang; amps_d[md] = axd["amp"]; md += 1
        if axq0:
            axq = quadrupole_axis_from_beta(beta, qstart)
            if axq:
                vq = axq["vec"] / np.linalg.norm(axq["vec"])
                ang = angular_distance(axq0["vec"] / np.linalg.norm(axq0["vec"]), vq)
                if ang is not None: angs_q[mq] = ang; eigs_q[mq] = axq["eigval"]; mq += 1
    out_d = out_q = None
    if md > 0:
        a_d, am_d = angs_d[:md], amps_d[:md]
        out_d = {"r68_deg": float(np.percentile(a_d, 68)), "r95_deg": float(np.percentile(a_d, 95)),
                 "amp_p50": float(np.percentile(am_d, 50)), "amp_p16": float(np.percentile(am_d, 16)),
                 "amp_p84": float(np.percentile(am_d, 84))}
    if mq > 0:
        a_q, e_q = angs_q[:mq], eigs_q[:mq]
        out_q = {"r68_deg": float(np.percentile(a_q, 68)), "r95_deg": float(np.percentile(a_q, 95)),
                 "eig_p50": float(np.percentile(e_q, 50)), "eig_p16": float(np.percentile(e_q, 16)),
                 "eig_p84": float(np.percentile(e_q, 84))}
    return {"dipole": out_d, "quadrupole": out_q}

# ========================= NULL CALIBRATION (VECTORIZED) =========================
def null_calibration_vectorized(yw_data, caches, beta_null, nsim=NSIM_NULL, seed=1234):
    rng = np.random.default_rng(seed)
    chi2_data = {n: fit_from_cache_yw(yw_data, c)[1] for n, c in caches.items()}
    chi2_mono = chi2_data["monopole"]
    dchi2_obs = {n: chi2_mono - chi2_data[n] for n in caches if n != "monopole"}
    null_dchi2 = {n: np.empty(nsim) for n in dchi2_obs}
    mean_null = caches["monopole"]["Mw"] @ beta_null
    npts = len(yw_data)
    chunk = min(10000, nsim)
    for start in range(0, nsim, chunk):
        end = min(start + chunk, nsim)
        yw_mock_T = (mean_null + rng.standard_normal((end - start, npts))).T
        mono_chi2 = np.sum((yw_mock_T - caches["monopole"]["Mw"] @ (caches["monopole"]["Mw_pinv"] @ yw_mock_T))**2, axis=0)
        for name in null_dchi2:
            model_chi2 = np.sum((yw_mock_T - caches[name]["Mw"] @ (caches[name]["Mw_pinv"] @ yw_mock_T))**2, axis=0)
            null_dchi2[name][start:end] = mono_chi2 - model_chi2
    out = {}
    for name, vals in null_dchi2.items():
        obs = dchi2_obs[name]
        pval = float(empirical_p(vals, obs))
        out[name] = {"observed_dchi2": float(obs), "empirical_p": pval,
                     "bonf_corrected_p": min(1.0, pval * N_Z_BINS),
                     "q95": float(np.percentile(vals, 95)), "q99": float(np.percentile(vals, 99))}
    return out

# ========================= ANALYSIS SAMPLE =========================
def analyze_sample(name, idx, data, Cfull, seed=123):
    rows = [data[i] for i in idx]
    z = np.array([r["z"] for r in rows]); y = np.array([r["mu"] for r in rows])
    C = Cfull[np.ix_(idx, idx)]; n = len(y)
    mu_iso = mu_iso_lcdm(z)
    nvec = sky_unit_vectors([r["ra"] for r in rows], [r["dec"] for r in rows])
    L, jitter = prepare_whitening(C)
    yw_data = np.linalg.solve(L, y)
    designs = build_joint_designs(mu_iso, nvec)
    caches = {m: build_model_cache(M, L) for m, M in designs.items()}
    fits = {m: fit_from_cache_yw(yw_data, caches[m]) for m in caches}
    chi2_mono, beta_mono = fits["monopole"][1], fits["monopole"][0]
    print(f"\n=== Sample: {name} | N={n} ===")
    print(f"  Baseline χ²={chi2_mono:.6f}, a={beta_mono[0]:.8f}, b={beta_mono[1]:.8f}, jitter={jitter}")
    res = {"sample": name, "N": int(n),
           "baseline_isotropic": {"chi2": float(chi2_mono),
                                  "params": [float(beta_mono[0]), float(beta_mono[1])],
                                  "jitter": float(jitter)},
           "residual_models": {}}
    mono_entry = {"chi2": float(chi2_mono), "AIC": float(aic(chi2_mono, 2)),
                  "BIC": float(bic(chi2_mono, 2, n)), "params": [float(v) for v in beta_mono]}
    # dipole
    beta_d, chi2_d = fits["dipole"]; kd = 5; dchi2_d = chi2_mono - chi2_d; dof_d = 3
    p_d = chi2_sf(dchi2_d, dof_d)
    ax_d = dipole_axis_from_beta(beta_d, 2)
    unc_d = axis_uncertainty_bootstrap(caches["dipole"], beta_d, 2, False, NSIM_AXIS, np.random.default_rng(seed+100))
    dip_entry = {"chi2": float(chi2_d), "AIC": float(aic(chi2_d, kd)), "BIC": float(bic(chi2_d, kd, n)),
                 "params": [float(v) for v in beta_d],
                 "delta_chi2": float(dchi2_d), "dof": int(dof_d), "p_analytic": float(p_d),
                 "delta_AIC": float(aic(chi2_d, kd) - aic(chi2_mono, 2)),
                 "delta_BIC": float(bic(chi2_d, kd, n) - bic(chi2_mono, 2, n)),
                 "bayes_factor_vs_mono": float(np.exp((bic(chi2_mono, 2, n) - bic(chi2_d, kd, n)) / 2))}
    if ax_d:
        dip_entry["axis"] = {"ra_deg": ax_d["ra_deg"], "dec_deg": ax_d["dec_deg"], "amp": ax_d["amp"],
                             "r68_deg": unc_d["r68_deg"] if unc_d else None,
                             "r95_deg": unc_d["r95_deg"] if unc_d else None,
                             "quality": axis_reliability(p_d, unc_d["r95_deg"] if unc_d else None)}
    # quadrupole
    beta_q, chi2_q = fits["quadrupole"]; kq = 7; dchi2_q = chi2_mono - chi2_q; dof_q = 5
    p_q = chi2_sf(dchi2_q, dof_q)
    ax_q = quadrupole_axis_from_beta(beta_q, 2)
    unc_q = axis_uncertainty_bootstrap(caches["quadrupole"], beta_q, 2, True, NSIM_AXIS, np.random.default_rng(seed+200))
    quad_entry = {"chi2": float(chi2_q), "AIC": float(aic(chi2_q, kq)), "BIC": float(bic(chi2_q, kq, n)),
                  "params": [float(v) for v in beta_q],
                  "delta_chi2": float(dchi2_q), "dof": int(dof_q), "p_analytic": float(p_q),
                  "delta_AIC": float(aic(chi2_q, kq) - aic(chi2_mono, 2)),
                  "delta_BIC": float(bic(chi2_q, kq, n) - bic(chi2_mono, 2, n)),
                  "bayes_factor_vs_mono": float(np.exp((bic(chi2_mono, 2, n) - bic(chi2_q, kq, n)) / 2))}
    if ax_q:
        quad_entry["axis"] = {"ra_deg": ax_q["ra_deg"], "dec_deg": ax_q["dec_deg"], "eigval": ax_q["eigval"],
                              "r68_deg": unc_q["r68_deg"] if unc_q else None,
                              "r95_deg": unc_q["r95_deg"] if unc_q else None,
                              "quality": axis_reliability(p_q, unc_q["r95_deg"] if unc_q else None)}
    # joint
    beta_j, chi2_j = fits["dipole+quadrupole"]; kj = 10; dchi2_j = chi2_mono - chi2_j; dof_j = 8
    p_j = chi2_sf(dchi2_j, dof_j)
    ax_jd = dipole_axis_from_beta(beta_j, 2)
    ax_jq = quadrupole_axis_from_beta(beta_j, 5)
    unc_j = joint_axis_uncertainty(caches["dipole+quadrupole"], beta_j, 2, 5, NSIM_AXIS, np.random.default_rng(seed+300))
    joint_entry = {"chi2": float(chi2_j), "AIC": float(aic(chi2_j, kj)), "BIC": float(bic(chi2_j, kj, n)),
                   "params": [float(v) for v in beta_j],
                   "delta_chi2": float(dchi2_j), "dof": int(dof_j), "p_analytic": float(p_j),
                   "delta_AIC": float(aic(chi2_j, kj) - aic(chi2_mono, 2)),
                   "delta_BIC": float(bic(chi2_j, kj, n) - bic(chi2_mono, 2, n)),
                   "bayes_factor_vs_mono": float(np.exp((bic(chi2_mono, 2, n) - bic(chi2_j, kj, n)) / 2))}
    if ax_jd:
        ud = unc_j["dipole"] if unc_j else None
        joint_entry["dipole_axis"] = {"ra_deg": ax_jd["ra_deg"], "dec_deg": ax_jd["dec_deg"], "amp": ax_jd["amp"],
                                      "r68_deg": ud["r68_deg"] if ud else None,
                                      "r95_deg": ud["r95_deg"] if ud else None,
                                      "quality": axis_reliability(p_j, ud["r95_deg"] if ud else None)}
    if ax_jq:
        uq = unc_j["quadrupole"] if unc_j else None
        joint_entry["quadrupole_axis"] = {"ra_deg": ax_jq["ra_deg"], "dec_deg": ax_jq["dec_deg"], "eigval": ax_jq["eigval"],
                                          "r68_deg": uq["r68_deg"] if uq else None,
                                          "r95_deg": uq["r95_deg"] if uq else None,
                                          "quality": axis_reliability(p_j, uq["r95_deg"] if uq else None)}
    if ax_d and ax_jd:
        joint_entry["alignment_dipole_vs_joint"] = angular_distance(ax_d["vec"], ax_jd["vec"])
    res["residual_models"]["monopole"] = mono_entry
    res["residual_models"]["dipole"] = dip_entry
    res["residual_models"]["quadrupole"] = quad_entry
    res["residual_models"]["dipole+quadrupole"] = joint_entry
    print("  Running null calibration...")
    null_res = null_calibration_vectorized(yw_data, caches, beta_mono, NSIM_NULL, seed+5000)
    res["null_calibration"] = null_res
    for m, r in null_res.items():
        print(f"    {m:18s} p_emp={r['empirical_p']:.6g} (bonf={r['bonf_corrected_p']:.6g}) q95={r['q95']:.3f} q99={r['q99']:.3f}")
    Tw = np.linalg.solve(L, quadrupole_basis_traceless(nvec))
    F_stf = Tw.T @ Tw
    svals = np.linalg.svd(F_stf, compute_uv=False)
    res["fisher_stf_svd"] = {"singular_values": svals.tolist(), "condition_number": float(svals[0]/svals[-1])}
    cov = np.linalg.inv(caches["dipole+quadrupole"]["Mw"].T @ caches["dipole+quadrupole"]["Mw"])
    cov_sub = cov[2:, 2:]
    std = np.sqrt(np.diag(cov_sub))
    corr = cov_sub / np.outer(std, std)
    res["parameter_correlation"] = corr.tolist()
    return res

# ========================= INJECTION, HEMISPHERE, JACKKNIFE, CMB =========================
def run_injection_tests(data_full, Cfull, orig_indices, idx_clean, nreal=NSIM_INJ, seed=999):
    idx = [orig_indices[k] for k in idx_clean]
    z = np.array([data_full[i]["z"] for i in idx]); y = np.array([data_full[i]["mu"] for i in idx])
    C = Cfull[np.ix_(idx, idx)]; n = len(z)
    mu_iso = mu_iso_lcdm(z); nvec = sky_unit_vectors([data_full[i]["ra"] for i in idx], [data_full[i]["dec"] for i in idx])
    qb = quadrupole_basis_traceless(nvec)
    M_dip = np.column_stack([np.ones(n), mu_iso, nvec])
    L, _ = prepare_whitening(C)
    Mw = np.linalg.solve(L, M_dip); Tw = np.linalg.solve(L, qb)
    R_dip = (np.linalg.inv(Mw.T @ Mw) @ (Mw.T @ Tw))[2:5, :]
    cache_dip = build_model_cache(M_dip, L)
    rng = np.random.default_rng(seed)
    all_results = []
    for mode_idx in range(5):
        s_inj = np.zeros(5); s_inj[mode_idx] = 0.1
        delta_mu = qb @ s_inj
        pred = R_dip @ s_inj
        recovered = np.empty((nreal, 3))
        for i in range(nreal):
            noise = L @ rng.standard_normal(n)
            y_tot = noise + delta_mu; yw_tot = np.linalg.solve(L, y_tot)
            beta, _ = fit_from_cache_yw(yw_tot, cache_dip); recovered[i] = beta[2:5]
        mean_rec = recovered.mean(axis=0)
        angle = angular_distance(pred, mean_rec)
        all_results.append({"mode_idx": mode_idx, "injected_s_vector": s_inj.tolist(),
                            "predicted_dipole": pred.tolist(), "recovered_mean": mean_rec.tolist(),
                            "recovered_std": recovered.std(axis=0).tolist(),
                            "angle_deg": angle, "amp_pred": float(np.linalg.norm(pred)),
                            "amp_rec": float(np.linalg.norm(mean_rec))})
    return all_results, R_dip

def hemisphere_asymmetry_test(data, Cfull, idx_orig, nsim_null=NSIM_HEMI, seed=777):
    rows = [data[i] for i in idx_orig]
    z = np.array([r["z"] for r in rows]); y = np.array([r["mu"] for r in rows])
    C = Cfull[np.ix_(idx_orig, idx_orig)]; n = len(y)
    mu_iso = mu_iso_lcdm(z)
    nvec = sky_unit_vectors([r["ra"] for r in rows], [r["dec"] for r in rows])
    L, _ = prepare_whitening(C); yw = np.linalg.solve(L, y)
    M_mono = np.column_stack([np.ones(n), mu_iso])
    cache_mono = build_model_cache(M_mono, L)
    beta_mono, chi2_mono = fit_from_cache_yw(yw, cache_mono)
    ra = np.array([r["ra"] for r in rows]); dec = np.array([r["dec"] for r in rows])
    a_G, d_G = np.radians(192.85948), np.radians(27.12825)
    ra_rad, dec_rad = np.radians(ra), np.radians(dec)
    sin_b = np.sin(d_G)*np.sin(dec_rad) + np.cos(d_G)*np.cos(dec_rad)*np.cos(ra_rad - a_G)
    b_deg = np.degrees(np.arcsin(sin_b))
    mask_north = b_deg > 0; mask_south = ~mask_north
    idx_n = np.where(mask_north)[0]; idx_s = np.where(mask_south)[0]
    def fit_hemisphere(idx_h):
        subC = Cfull[np.ix_(idx_h, idx_h)]
        z_h = z[idx_h]; y_h = y[idx_h]; mu_h = mu_iso_lcdm(z_h)
        M_h = np.column_stack([np.ones_like(mu_h), mu_h])
        L_h, _ = prepare_whitening(subC); yw_h = np.linalg.solve(L_h, y_h)
        cache_h = build_model_cache(M_h, L_h)
        beta_h, chi2_h = fit_from_cache_yw(yw_h, cache_h)
        return beta_h, chi2_h, len(z_h)
    b_n, chi2_n, n_n = fit_hemisphere(idx_n)
    b_s, chi2_s, n_s = fit_hemisphere(idx_s)
    chi2_combined = chi2_n + chi2_s
    dchi2 = chi2_mono - chi2_combined
    p_hemi = chi2_sf(dchi2, 2)
    rng = np.random.default_rng(seed)
    null_dchi2 = np.empty(nsim_null)
    for i in range(nsim_null):
        perm = rng.permutation(len(z))
        idx_n_rand = perm[:n_n]; idx_s_rand = perm[n_n:]
        _, c2_n, _ = fit_hemisphere(idx_n_rand)
        _, c2_s, _ = fit_hemisphere(idx_s_rand)
        null_dchi2[i] = chi2_mono - (c2_n + c2_s)
    p_emp = empirical_p(null_dchi2, dchi2)
    print(f"  Galactic north (N={n_n}): χ²={chi2_n:.3f}, a={b_n[0]:.4f}, b={b_n[1]:.4f}")
    print(f"  Galactic south (N={n_s}): χ²={chi2_s:.3f}, a={b_s[0]:.4f}, b={b_s[1]:.4f}")
    print(f"  Combined χ²={chi2_combined:.3f}, Δχ²={dchi2:.3f}, p={p_hemi:.4f}")
    print(f"  Empirical p (shuffle): {p_emp:.4f}")
    return {"hemisphere": "Galactic north/south", "N_north": int(n_n), "N_south": int(n_s),
            "chi2_north": float(chi2_n), "chi2_south": float(chi2_s), "delta_chi2": float(dchi2),
            "p_analytic": float(p_hemi), "p_empirical": float(p_emp)}

def jackknife_dipole_stability(data, Cfull, idx_orig, n_removals=100, frac_remove=JACK_FRAC_REMOVE, seed=888):
    rows = [data[i] for i in idx_orig]
    z_all = np.array([r["z"] for r in rows]); y_all = np.array([r["mu"] for r in rows])
    C = Cfull[np.ix_(idx_orig, idx_orig)]
    mu_iso_all = mu_iso_lcdm(z_all)
    nvec_all = sky_unit_vectors([r["ra"] for r in rows], [r["dec"] for r in rows])
    L, _ = prepare_whitening(C)
    M_dip = np.column_stack([np.ones_like(mu_iso_all), mu_iso_all, nvec_all])
    cache_dip = build_model_cache(M_dip, L)
    yw_all = np.linalg.solve(L, y_all)
    beta_full, _ = fit_from_cache_yw(yw_all, cache_dip)
    ref_ax = dipole_axis_from_beta(beta_full, 2)
    ref_dir = ref_ax["vec"] / np.linalg.norm(ref_ax["vec"])
    n_total = len(z_all)
    rng = np.random.default_rng(seed)
    angles = []
    for _ in range(n_removals):
        n_remove = max(1, int(frac_remove * n_total))
        idx_keep = np.sort(rng.choice(n_total, size=n_total - n_remove, replace=False))
        z_sub = z_all[idx_keep]; y_sub = y_all[idx_keep]
        C_sub = C[np.ix_(idx_keep, idx_keep)]
        mu_iso_sub = mu_iso_all[idx_keep]; nvec_sub = nvec_all[idx_keep]
        try:
            L_sub, _ = prepare_whitening(C_sub)
            M_dip_sub = np.column_stack([np.ones_like(mu_iso_sub), mu_iso_sub, nvec_sub])
            cache_sub = build_model_cache(M_dip_sub, L_sub)
            yw_sub = np.linalg.solve(L_sub, y_sub)
            beta_sub, _ = fit_from_cache_yw(yw_sub, cache_sub)
            ax_sub = dipole_axis_from_beta(beta_sub, 2)
            dir_sub = ax_sub["vec"] / np.linalg.norm(ax_sub["vec"])
            angle = angular_distance(ref_dir, dir_sub)
            if angle is not None: angles.append(angle)
        except np.linalg.LinAlgError: continue
    if len(angles) == 0: return None
    angles = np.array(angles)
    return {"mean_angle_deg": float(np.mean(angles)), "std_angle_deg": float(np.std(angles)),
            "rms_angle_deg": float(np.sqrt(np.mean(angles**2))), "nsuccess": int(len(angles))}

def cmb_alignment_test(dipole_vec, cmb_ra=CMB_DIPOLE_RA, cmb_dec=CMB_DIPOLE_DEC, nsim=5000, seed=999):
    cmb_vec = sky_unit_vectors([cmb_ra], [cmb_dec])[0]
    angle = angular_distance(dipole_vec, cmb_vec)
    rng = np.random.default_rng(seed)
    rand_angles = np.empty(nsim)
    for i in range(nsim):
        rand_v = rng.standard_normal(3); rand_v /= np.linalg.norm(rand_v)
        rand_angles[i] = angular_distance(rand_v, cmb_vec)
    p_value = empirical_p(rand_angles, angle)
    return {"angle_deg": angle, "p_value": p_value}

# ========================= BULK FLOW FIT =========================
def bulk_flow_fit(z, y, nvec, C):
    n = len(z)
    mu_iso = mu_iso_lcdm(z)
    M_mono = np.column_stack([np.ones(n), mu_iso])
    prefactor = (5.0 / math.log(10)) / C_LIGHT
    vel_design = nvec * (prefactor / z[:, None])
    M = np.column_stack([M_mono, vel_design])
    L, _ = prepare_whitening(C)
    yw = np.linalg.solve(L, y)
    cache = build_model_cache(M, L)
    beta, chi2 = fit_from_cache_yw(yw, cache)
    v_vec = beta[2:5]
    amp = np.linalg.norm(v_vec)
    ra, dec = vec_to_radec(v_vec)
    cache_mono = build_model_cache(M_mono, L)
    _, chi2_mono = fit_from_cache_yw(yw, cache_mono)
    dchi2 = chi2_mono - chi2
    p_val = chi2_sf(dchi2, 3)
    return {"v_amp_km_s": amp, "ra_deg": ra, "dec_deg": dec,
            "chi2": chi2, "delta_chi2": dchi2, "p_value": p_val, "v_vec": v_vec}

# ========================= CUMULATIVE TOMOGRAPHY (DIPOLE, GLOBAL) =========================
def cumulative_tomography_global(data_full, Cfull_all, orig_indices, z_clean, z_max_list=None,
                                 nsim_null=NSIM_TOMO_NULL, seed=42, output_csv=OUTPUT_TOMO):
    if z_max_list is None:
        z_max_list = np.linspace(0.02, 0.30, 40)
    tomo_data = []
    for zmax in z_max_list:
        mask = z_clean <= zmax
        idx_clean = np.where(mask)[0]
        if len(idx_clean) < 30: continue
        idx_orig = [orig_indices[k] for k in idx_clean]
        rows = [data_full[i] for i in idx_orig]
        z = np.array([r["z"] for r in rows]); y = np.array([r["mu"] for r in rows])
        Csub = Cfull_all[np.ix_(idx_orig, idx_orig)]; n = len(z)
        mu_iso = mu_iso_lcdm(z)
        nvec = sky_unit_vectors([r["ra"] for r in rows], [r["dec"] for r in rows])
        try:
            L, _ = prepare_whitening(Csub)
            M_mono = np.column_stack([np.ones(n), mu_iso])
            M_dip = np.column_stack([M_mono, nvec])
            yw = np.linalg.solve(L, y)
            cache_mono = build_model_cache(M_mono, L)
            beta_mono, chi2_mono = fit_from_cache_yw(yw, cache_mono)
            cache_dip = build_model_cache(M_dip, L)
            beta_dip, chi2_dip = fit_from_cache_yw(yw, cache_dip)
            dchi2 = chi2_mono - chi2_dip
            dip_vec = beta_dip[2:5]; amp = np.linalg.norm(dip_vec)
            ra, dec = vec_to_radec(dip_vec)
            from math import radians, sin, cos, asin, atan2
            a_G,d_G,l_NCP = radians(192.85948), radians(27.12825), radians(122.932)
            ra_r,dec_r = radians(ra), radians(dec)
            sin_b = sin(d_G)*sin(dec_r)+cos(d_G)*cos(dec_r)*cos(ra_r-a_G)
            b_gal = asin(sin_b)
            cos_b_sin_l = cos(dec_r)*sin(ra_r-a_G)
            cos_b_cos_l = cos(d_G)*sin(dec_r)-sin(d_G)*cos(dec_r)*cos(ra_r-a_G)
            l_gal = l_NCP - atan2(cos_b_sin_l, cos_b_cos_l)
            l_deg = np.degrees(l_gal)%360; b_deg = np.degrees(b_gal)
            tomo_data.append({
                "z_max": zmax, "N": n, "delta_chi2": dchi2,
                "p_analytic": chi2_sf(dchi2,3), "amplitude": amp,
                "l_deg": l_deg, "b_deg": b_deg,
                "idx_orig": idx_orig, "L": L, "cache_mono": cache_mono, "cache_dip": cache_dip
            })
        except: continue
    if not tomo_data: return None
    obs_max = max(r["delta_chi2"] for r in tomo_data)
    full_idx_orig = [orig_indices[k] for k in range(len(z_clean))]
    z_full = np.array([data_full[i]["z"] for i in full_idx_orig])
    y_full = np.array([data_full[i]["mu"] for i in full_idx_orig])
    C_full = Cfull_all[np.ix_(full_idx_orig, full_idx_orig)]
    mu_iso_full = mu_iso_lcdm(z_full)
    M_base_full = np.column_stack([np.ones_like(mu_iso_full), mu_iso_full])
    L_full, _ = prepare_whitening(C_full)
    yw_full = np.linalg.solve(L_full, y_full)
    cache_mono_full = build_model_cache(M_base_full, L_full)
    beta_mono_full, _ = fit_from_cache_yw(yw_full, cache_mono_full)
    mean_yw_full = cache_mono_full["Mw"] @ beta_mono_full
    n_full = len(z_full)
    rng = np.random.default_rng(seed)
    max_null = np.empty(nsim_null)
    for i_sim in range(nsim_null):
        yw_mock_full = mean_yw_full + rng.standard_normal(n_full)
        sim_max = 0.0
        for rec in tomo_data:
            sub_idx = [full_idx_orig.index(k) for k in rec["idx_orig"]]
            yw_sub = yw_mock_full[sub_idx]
            try:
                _, chi2_dip = fit_from_cache_yw(yw_sub, rec["cache_dip"])
                _, chi2_mono = fit_from_cache_yw(yw_sub, rec["cache_mono"])
                dchi2 = chi2_mono - chi2_dip
                if dchi2 > sim_max: sim_max = dchi2
            except: continue
        max_null[i_sim] = sim_max
        if (i_sim+1)%500==0: print(f"    tomography null sim {i_sim+1}/{nsim_null}")
    p_global_tomo = empirical_p(max_null, obs_max)
    csv_rows = [{k: r[k] for k in ["z_max","N","delta_chi2","p_analytic","amplitude","l_deg","b_deg"]} for r in tomo_data]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["z_max","N","delta_chi2","p_analytic","amplitude","l_deg","b_deg"])
        w.writeheader(); w.writerows(csv_rows)
    print(f"Cumulative tomography saved. Global p = {p_global_tomo:.4f}")
    return {"z_max": [r["z_max"] for r in tomo_data], "delta_chi2": [r["delta_chi2"] for r in tomo_data],
            "amplitude": [r["amplitude"] for r in tomo_data], "l_deg": [r["l_deg"] for r in tomo_data],
            "b_deg": [r["b_deg"] for r in tomo_data], "global_max_dchi2": obs_max,
            "global_p_value": p_global_tomo, "nsim_tomo_null": nsim_null}

# ========================= BULK FLOW TOMOGRAPHY =========================
def cumulative_bulkflow_tomography(data_full, Cfull_all, orig_indices, z_clean, z_max_list=None, output_csv=OUTPUT_BULK_TOMO):
    if z_max_list is None:
        z_max_list = np.linspace(0.02, 0.30, 40)
    results = []
    for zmax in z_max_list:
        mask = z_clean <= zmax
        idx_clean = np.where(mask)[0]
        if len(idx_clean) < 30: continue
        idx_orig = [orig_indices[k] for k in idx_clean]
        rows = [data_full[i] for i in idx_orig]
        z = np.array([r["z"] for r in rows]); y = np.array([r["mu"] for r in rows])
        Csub = Cfull_all[np.ix_(idx_orig, idx_orig)]
        nvec = sky_unit_vectors([r["ra"] for r in rows], [r["dec"] for r in rows])
        try:
            bf = bulk_flow_fit(z, y, nvec, Csub)
            results.append({"z_max": zmax, "N": len(z),
                            "v_amp_km_s": bf["v_amp_km_s"],
                            "ra_deg": bf["ra_deg"], "dec_deg": bf["dec_deg"],
                            "delta_chi2": bf["delta_chi2"], "p_analytic": bf["p_value"]})
        except np.linalg.LinAlgError: continue
    if not results: return None
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["z_max","N","v_amp_km_s","ra_deg","dec_deg","delta_chi2","p_analytic"])
        w.writeheader(); w.writerows(results)
    print(f"Bulk flow tomography saved to {output_csv}")
    return results

# ========================= PROFILE SKY MAP =========================
def profile_sky_map(z, y, nvec, C, nside_ra=36, nside_dec=18, output_csv=OUTPUT_SKY_MAP):
    n = len(z); mu_iso = mu_iso_lcdm(z)
    L, _ = prepare_whitening(C); yw = np.linalg.solve(L, y)
    M_mono = np.column_stack([np.ones(n), mu_iso])
    cache_mono = build_model_cache(M_mono, L)
    _, chi2_mono = fit_from_cache_yw(yw, cache_mono)
    ra_grid = np.linspace(0, 360, nside_ra+1)[:-1]
    dec_grid = np.linspace(-90, 90, nside_dec+1)[:-1]
    out_rows = []
    for ra in ra_grid:
        for dec in dec_grid:
            dir_vec = sky_unit_vectors([ra], [dec])[0]
            dip_comp = nvec @ dir_vec
            M = np.column_stack([M_mono, dip_comp])
            cache = build_model_cache(M, L)
            _, chi2 = fit_from_cache_yw(yw, cache)
            dchi2 = chi2_mono - chi2
            out_rows.append({"ra": ra, "dec": dec, "delta_chi2": dchi2})
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ra","dec","delta_chi2"])
        w.writeheader(); w.writerows(out_rows)
    print(f"Profile sky map saved to {output_csv}")
    return out_rows

# ========================= SURVEY SPLIT ANALYSIS =========================
def survey_split_analysis(data_full, Cfull_all, orig_indices, z_clean, samples_def):
    if not any('survey' in d for d in data_full):
        return None
    idx_clean = samples_def["z≤0.03"]
    surveys = {}
    for i in idx_clean:
        s = data_full[orig_indices[i]].get('survey','unknown')
        surveys.setdefault(s, []).append(i)
    per_survey = {}
    for surv, idx_sub in surveys.items():
        if len(idx_sub) < 30: continue
        idx_orig_sub = [orig_indices[k] for k in idx_sub]
        try:
            z = np.array([data_full[i]["z"] for i in idx_orig_sub])
            y = np.array([data_full[i]["mu"] for i in idx_orig_sub])
            C = Cfull_all[np.ix_(idx_orig_sub, idx_orig_sub)]
            mu_iso = mu_iso_lcdm(z)
            nvec = sky_unit_vectors([data_full[i]["ra"] for i in idx_orig_sub],
                                    [data_full[i]["dec"] for i in idx_orig_sub])
            L, _ = prepare_whitening(C)
            M_mono = np.column_stack([np.ones_like(mu_iso), mu_iso])
            M_dip = np.column_stack([M_mono, nvec])
            yw = np.linalg.solve(L, y)
            cache_mono = build_model_cache(M_mono, L)
            _, chi2_mono = fit_from_cache_yw(yw, cache_mono)
            cache_dip = build_model_cache(M_dip, L)
            beta_dip, chi2_dip = fit_from_cache_yw(yw, cache_dip)
            dchi2 = chi2_mono - chi2_dip
            ax = dipole_axis_from_beta(beta_dip)
            per_survey[surv] = {"N":len(z), "delta_chi2": dchi2, "p": chi2_sf(dchi2,3),
                                "ra": ax["ra_deg"], "dec": ax["dec_deg"], "amp": ax["amp"]}
        except Exception as e: per_survey[surv] = {"error": str(e)}
    loo = {}
    all_surveys = list(surveys.keys())
    for surv_out in all_surveys:
        idx_keep = [i for s in all_surveys if s != surv_out for i in surveys[s]]
        if len(idx_keep) < 50: continue
        idx_orig_keep = [orig_indices[k] for k in idx_keep]
        try:
            z = np.array([data_full[i]["z"] for i in idx_orig_keep])
            y = np.array([data_full[i]["mu"] for i in idx_orig_keep])
            C = Cfull_all[np.ix_(idx_orig_keep, idx_orig_keep)]
            mu_iso = mu_iso_lcdm(z)
            nvec = sky_unit_vectors([data_full[i]["ra"] for i in idx_orig_keep],
                                    [data_full[i]["dec"] for i in idx_orig_keep])
            L, _ = prepare_whitening(C)
            M_dip = np.column_stack([np.ones_like(mu_iso), mu_iso, nvec])
            yw = np.linalg.solve(L, y)
            cache_dip = build_model_cache(M_dip, L)
            beta_dip, _ = fit_from_cache_yw(yw, cache_dip)
            ax = dipole_axis_from_beta(beta_dip)
            loo[surv_out] = {"ra": ax["ra_deg"], "dec": ax["dec_deg"], "amp": ax["amp"]}
        except Exception as e: loo[surv_out] = {"error": str(e)}
    with open(OUTPUT_SURVEY, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Survey","N","delta_chi2","p","RA","Dec","Amplitude"])
        for surv, info in per_survey.items():
            if "error" in info:
                w.writerow([surv, "", "", "", "", "", info["error"]])
            else:
                w.writerow([surv, info["N"], info["delta_chi2"], info["p"],
                            info["ra"], info["dec"], info["amp"]])
        w.writerow([]); w.writerow(["Leave-one-out jackknife"])
        w.writerow(["Removed Survey","RA","Dec","Amplitude"])
        for surv, info in loo.items():
            if "error" in info:
                w.writerow([surv, "", "", info["error"]])
            else:
                w.writerow([surv, info["ra"], info["dec"], info["amp"]])
    print(f"Survey split results saved to {OUTPUT_SURVEY}")
    return {"per_survey": per_survey, "leave_one_out": loo}

# ========================= SHAPLEY CONE REMOVAL TEST =========================
def shapley_cone_removal_test(data_full, Cfull_all, orig_indices, z_clean, samples_def,
                              cone_angles=[20,30,40,60], output_csv=OUTPUT_SHAPLEY):
    idx_clean = samples_def["z≤0.03"]
    idx_orig = [orig_indices[k] for k in idx_clean]
    rows = [data_full[i] for i in idx_orig]
    z_all = np.array([r["z"] for r in rows])
    y_all = np.array([r["mu"] for r in rows])
    nvec_all = sky_unit_vectors([r["ra"] for r in rows], [r["dec"] for r in rows])
    C_all = Cfull_all[np.ix_(idx_orig, idx_orig)]
    shapley_vec = sky_unit_vectors([SHAPLEY_RA], [SHAPLEY_DEC])[0]

    # Full sample baseline
    full_dip = None; full_bulk = None
    try:
        mu_iso = mu_iso_lcdm(z_all); n = len(z_all)
        L, _ = prepare_whitening(C_all)
        M_mono = np.column_stack([np.ones(n), mu_iso])
        M_dip = np.column_stack([M_mono, nvec_all])
        yw = np.linalg.solve(L, y_all)
        cache_mono = build_model_cache(M_mono, L)
        _, chi2_mono = fit_from_cache_yw(yw, cache_mono)
        cache_dip = build_model_cache(M_dip, L)
        beta_dip, chi2_dip = fit_from_cache_yw(yw, cache_dip)
        dchi2 = chi2_mono - chi2_dip
        ax_dip = dipole_axis_from_beta(beta_dip)
        full_dip = {"delta_chi2": dchi2, "amp": ax_dip["amp"],
                    "ra": ax_dip["ra_deg"], "dec": ax_dip["dec_deg"], "p": chi2_sf(dchi2,3)}
        bf = bulk_flow_fit(z_all, y_all, nvec_all, C_all)
        full_bulk = {"v_amp": bf["v_amp_km_s"], "ra": bf["ra_deg"], "dec": bf["dec_deg"],
                     "delta_chi2": bf["delta_chi2"], "p": bf["p_value"]}
    except Exception as e:
        print(f"Shapley cone test: full sample fit failed: {e}")
        return None

    results = []
    for angle in cone_angles:
        dists = np.array([angular_distance(nv, shapley_vec) for nv in nvec_all])
        keep_mask = dists >= angle
        idx_keep = np.where(keep_mask)[0]
        if len(idx_keep) < 30: continue
        z_sub = z_all[idx_keep]; y_sub = y_all[idx_keep]
        nvec_sub = nvec_all[idx_keep]; C_sub = C_all[np.ix_(idx_keep, idx_keep)]
        try:
            n = len(z_sub); mu_iso = mu_iso_lcdm(z_sub)
            L, _ = prepare_whitening(C_sub)
            M_mono = np.column_stack([np.ones(n), mu_iso])
            M_dip = np.column_stack([M_mono, nvec_sub])
            yw = np.linalg.solve(L, y_sub)
            cache_mono = build_model_cache(M_mono, L)
            _, chi2_mono = fit_from_cache_yw(yw, cache_mono)
            cache_dip = build_model_cache(M_dip, L)
            beta_dip, chi2_dip = fit_from_cache_yw(yw, cache_dip)
            dchi2_dip = chi2_mono - chi2_dip
            ax_dip = dipole_axis_from_beta(beta_dip)
            bf = bulk_flow_fit(z_sub, y_sub, nvec_sub, C_sub)
            results.append({
                "cone_angle": angle,
                "N_removed": len(z_all) - len(idx_keep),
                "N_kept": len(idx_keep),
                "dipole_delta_chi2": dchi2_dip,
                "dipole_p": chi2_sf(dchi2_dip, 3),
                "dipole_amp": ax_dip["amp"],
                "dipole_ra": ax_dip["ra_deg"],
                "dipole_dec": ax_dip["dec_deg"],
                "bulk_amp": bf["v_amp_km_s"],
                "bulk_ra": bf["ra_deg"],
                "bulk_dec": bf["dec_deg"],
                "bulk_delta_chi2": bf["delta_chi2"],
                "bulk_p": bf["p_value"]
            })
        except Exception as e:
            results.append({"cone_angle": angle, "error": str(e)})

    # Build full row explicitly as a dict with all keys
    full_row = {
        "cone_angle": 0,
        "N_removed": 0,
        "N_kept": len(z_all),
        "dipole_delta_chi2": full_dip["delta_chi2"],
        "dipole_p": full_dip["p"],
        "dipole_amp": full_dip["amp"],
        "dipole_ra": full_dip["ra"],
        "dipole_dec": full_dip["dec"],
        "bulk_amp": full_bulk["v_amp"],
        "bulk_ra": full_bulk["ra"],
        "bulk_dec": full_bulk["dec"],
        "bulk_delta_chi2": full_bulk["delta_chi2"],
        "bulk_p": full_bulk["p"]
    }

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cone_angle","N_removed","N_kept",
                                          "dipole_delta_chi2","dipole_p","dipole_amp","dipole_ra","dipole_dec",
                                          "bulk_amp","bulk_ra","bulk_dec","bulk_delta_chi2","bulk_p"])
        w.writeheader()
        w.writerow(full_row)
        for row in results:
            w.writerow(row)

    # Print summary table
    print("\n--- Shapley Cone Removal Test Summary ---")
    print(f"{'Angle':<8} {'Removed':<8} {'Kept':<8} {'Dip Δχ²':<10} {'Dip p':<8} {'Dip Amp':<10} {'Dip RA':<8} {'Dip Dec':<8} {'Bulk v':<8} {'Bulk RA':<8} {'Bulk Dec':<8} {'Bulk Δχ²':<10} {'Bulk p':<8}")
    print("-"*100)
    all_rows = [full_row] + results
    for r in all_rows:
        if "error" in r:
            print(f"{r['cone_angle']:<8} {'ERROR':<8}")
        else:
            print(f"{r['cone_angle']:<8} {r['N_removed']:<8} {r['N_kept']:<8} "
                  f"{r['dipole_delta_chi2']:<10.2f} {r['dipole_p']:<8.4f} {r['dipole_amp']:<10.4f} "
                  f"{r['dipole_ra']:<8.2f} {r['dipole_dec']:<8.2f} "
                  f"{r['bulk_amp']:<8.1f} {r['bulk_ra']:<8.2f} {r['bulk_dec']:<8.2f} "
                  f"{r['bulk_delta_chi2']:<10.2f} {r['bulk_p']:<8.4f}")
    print(f"\nShapley cone removal test results saved to {output_csv}")
    return results

# ========================= COMPARISON WITH KNOWN STRUCTURES =========================
def compare_directions(vec_list, labels, structures=KNOWN_STRUCTURES):
    print("\n=== Comparison with Known Large‑Scale Structures ===")
    for label, v in zip(labels, vec_list):
        ra, dec = vec_to_radec(v)
        print(f"  {label}: RA={ra:.1f}, Dec={dec:.1f}")
        for sname, (sra, sdec) in structures.items():
            svec = sky_unit_vectors([sra], [sdec])[0]
            ang = angular_distance(v, svec)
            print(f"    vs {sname}: {ang:.1f}°")

# ========================= ALIGNMENT SUMMARY =========================
def compute_alignment_summary(results_all):
    samples = [k for k, v in results_all.items() if isinstance(v, dict) and "residual_models" in v]
    rows = []
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            d1 = results_all[samples[i]]["residual_models"].get("dipole",{}).get("axis",None)
            d2 = results_all[samples[j]]["residual_models"].get("dipole",{}).get("axis",None)
            if d1 and d2:
                v1 = np.array([np.cos(np.radians(d1["dec_deg"]))*np.cos(np.radians(d1["ra_deg"])),
                               np.cos(np.radians(d1["dec_deg"]))*np.sin(np.radians(d1["ra_deg"])),
                               np.sin(np.radians(d1["dec_deg"]))])
                v2 = np.array([np.cos(np.radians(d2["dec_deg"]))*np.cos(np.radians(d2["ra_deg"])),
                               np.cos(np.radians(d2["dec_deg"]))*np.sin(np.radians(d2["ra_deg"])),
                               np.sin(np.radians(d2["dec_deg"]))])
                ang = angular_distance(v1, v2)
                rows.append({"type":"dipole_vs_dipole","sample_1":samples[i],"sample_2":samples[j],
                             "angle_deg":ang,"quality_1":d1.get("quality",""),"quality_2":d2.get("quality","")})
    for s in samples:
        d1 = results_all[s]["residual_models"].get("dipole",{}).get("axis",None)
        d2 = results_all[s]["residual_models"].get("dipole+quadrupole",{}).get("dipole_axis",None)
        if d1 and d2:
            v1 = np.array([np.cos(np.radians(d1["dec_deg"]))*np.cos(np.radians(d1["ra_deg"])),
                           np.cos(np.radians(d1["dec_deg"]))*np.sin(np.radians(d1["ra_deg"])),
                           np.sin(np.radians(d1["dec_deg"]))])
            v2 = np.array([np.cos(np.radians(d2["dec_deg"]))*np.cos(np.radians(d2["ra_deg"])),
                           np.cos(np.radians(d2["dec_deg"]))*np.sin(np.radians(d2["ra_deg"])),
                           np.sin(np.radians(d2["dec_deg"]))])
            ang = angular_distance(v1, v2)
            rows.append({"type":"dipole_vs_jointdipole","sample_1":s,"sample_2":s,
                         "angle_deg":ang,"quality_1":d1.get("quality",""),"quality_2":d2.get("quality","")})
    return rows

# ========================= MAIN =========================
def main():
    t0 = time.time()
    print("Starting final integrated analysis...")
    data_full = load_catalog(CATALOG_FILE)
    nall = len(data_full)
    print(f"Loaded {nall} SNe")
    data_clean = [d for d in data_full if not d["is_ceph"]]
    nclean = len(data_clean)
    print(f"Removed {nall-nclean} calibrators → {nclean} SNe")
    orig_indices = [i for i,d in enumerate(data_full) if not d["is_ceph"]]
    Cfull = load_covariance(COV_FILE, nall)
    print(f"Covariance shape: {Cfull.shape}")
    z_clean = np.array([d["z"] for d in data_clean])
    samples_def = {
        "full": np.arange(nclean).tolist(),
        "z≤0.03": np.where(z_clean <= 0.03)[0].tolist(),
        "0.03<z≤0.10": np.where((z_clean > 0.03) & (z_clean <= 0.10))[0].tolist(),
        "0.10<z≤0.30": np.where((z_clean > 0.10) & (z_clean <= 0.30))[0].tolist(),
        "z>0.30": np.where(z_clean > 0.30)[0].tolist(),
    }
    results_all = {}
    base_seed = 20260624
    extra_info = {}
    for i, (name, idx_clean) in enumerate(samples_def.items()):
        idx_orig = [orig_indices[k] for k in idx_clean]
        sample_res = analyze_sample(name, idx_orig, data_full, Cfull, seed=base_seed+1000*i)
        results_all[name] = sample_res
        if name == "z≤0.03":
            extra_info["idx_orig"] = idx_orig

    # Injection tests
    print("\n=== Full injection tests (z≤0.03) ===")
    inj_results, R_dip = run_injection_tests(data_full, Cfull, orig_indices, samples_def["z≤0.03"], NSIM_INJ, 12345)
    results_all["injection_tests"] = inj_results
    for r in inj_results:
        print(f"  Mode {r['mode_idx']}: injected {r['injected_s_vector']} → predicted amp={r['amp_pred']:.4f}, recovered amp={r['amp_rec']:.4f}, angle={r['angle_deg']:.3f}°")
    np.savetxt(OUTPUT_RMAT, R_dip, delimiter=",", header="3x5 R_dip matrix")
    scatter = []
    for r in inj_results:
        for c, comp in enumerate(["x","y","z"]):
            scatter.append({"mode_idx":r["mode_idx"],"component":comp,"predicted":r["predicted_dipole"][c],"recovered":r["recovered_mean"][c]})
    with open(OUTPUT_SCATTER,"w",newline="") as f:
        w = csv.DictWriter(f, ["mode_idx","component","predicted","recovered"]); w.writeheader(); w.writerows(scatter)
    with open(OUTPUT_INJ_JSON,"w") as f: json.dump(make_json_serializable(inj_results), f, indent=2)

    # Bayes factor summary
    print("\n=== Bayes Factors (vs Monopole) ===")
    for sname, sres in results_all.items():
        if "residual_models" not in sres: continue
        for mname, mentry in sres["residual_models"].items():
            if "bayes_factor_vs_mono" in mentry:
                print(f"  {sname} {mname}: BF vs mono = {mentry['bayes_factor_vs_mono']:.3f}")

    # Hemisphere asymmetry test
    print("\n=== Hemisphere Asymmetry Test (z≤0.03) ===")
    hemi_res = hemisphere_asymmetry_test(data_full, Cfull, extra_info["idx_orig"], nsim_null=NSIM_HEMI)
    results_all["hemisphere_asymmetry"] = hemi_res

    # Jackknife
    print("\n=== Jackknife Dipole Stability (z≤0.03) ===")
    jack_res = jackknife_dipole_stability(data_full, Cfull, extra_info["idx_orig"], n_removals=100, frac_remove=JACK_FRAC_REMOVE, seed=42)
    if jack_res:
        print(f"  Mean angular deviation: {jack_res['mean_angle_deg']:.2f}° ± {jack_res['std_angle_deg']:.2f}° (rms={jack_res['rms_angle_deg']:.2f}°)")
        results_all["jackknife_dipole"] = jack_res

    # CMB alignment
    print("\n=== CMB Alignment (z≤0.03 dipole) ===")
    dipole_axis_info = results_all["z≤0.03"]["residual_models"]["dipole"].get("axis", None)
    if dipole_axis_info:
        v_dip = np.array([np.cos(np.radians(dipole_axis_info["dec_deg"]))*np.cos(np.radians(dipole_axis_info["ra_deg"])),
                          np.cos(np.radians(dipole_axis_info["dec_deg"]))*np.sin(np.radians(dipole_axis_info["ra_deg"])),
                          np.sin(np.radians(dipole_axis_info["dec_deg"]))])
        cmb_test = cmb_alignment_test(v_dip)
        print(f"  Angle to CMB dipole: {cmb_test['angle_deg']:.1f}°, p-value: {cmb_test['p_value']:.4f}")
        results_all["cmb_alignment"] = cmb_test

    # Cumulative Tomography (dipole)
    print("\n=== Cumulative Redshift Tomography (global calibration) ===")
    tomo_res = cumulative_tomography_global(data_full, Cfull, orig_indices, z_clean, nsim_null=NSIM_TOMO_NULL, seed=2026)
    if tomo_res:
        results_all["cumulative_tomography"] = tomo_res

    # Bulk flow fit for z≤0.03
    print("\n=== Bulk Flow Fit (z≤0.03) ===")
    idx_orig_z003 = [orig_indices[k] for k in samples_def["z≤0.03"]]
    rows_z003 = [data_full[i] for i in idx_orig_z003]
    z003 = np.array([r["z"] for r in rows_z003]); y003 = np.array([r["mu"] for r in rows_z003])
    C003 = Cfull[np.ix_(idx_orig_z003, idx_orig_z003)]
    nvec003 = sky_unit_vectors([r["ra"] for r in rows_z003], [r["dec"] for r in rows_z003])
    bulk_res = bulk_flow_fit(z003, y003, nvec003, C003)
    results_all["bulk_flow_z003"] = bulk_res
    print(f"  Bulk velocity: {bulk_res['v_amp_km_s']:.1f} km/s, direction RA={bulk_res['ra_deg']:.1f}°, Dec={bulk_res['dec_deg']:.1f}°, Δχ²={bulk_res['delta_chi2']:.2f}, p={bulk_res['p_value']:.4f}")

    # Bulk flow tomography
    print("\n=== Bulk Flow Cumulative Tomography ===")
    bulk_tomo = cumulative_bulkflow_tomography(data_full, Cfull, orig_indices, z_clean)
    if bulk_tomo:
        results_all["bulkflow_tomography"] = bulk_tomo

    # Profile likelihood sky map
    print("\n=== Profile Likelihood Sky Map (z≤0.03) ===")
    profile_sky_map(z003, y003, nvec003, C003, nside_ra=36, nside_dec=18)

    # Survey split analysis
    print("\n=== Survey Split Analysis ===")
    survey_res = survey_split_analysis(data_full, Cfull, orig_indices, z_clean, samples_def)
    if survey_res is not None:
        results_all["survey_analysis"] = survey_res

    # Shapley Cone Removal Test
    print("\n=== Shapley Cone Removal Test ===")
    shapley_cone_results = shapley_cone_removal_test(data_full, Cfull, orig_indices, z_clean, samples_def)
    if shapley_cone_results:
        results_all["shapley_cone_test"] = shapley_cone_results

    # Comparison with Known Structures
    print("\n=== Comparison with Known Structures ===")
    vec_bulk = bulk_res["v_vec"]
    vec_dipole = np.array([np.cos(np.radians(dipole_axis_info["dec_deg"]))*np.cos(np.radians(dipole_axis_info["ra_deg"])),
                           np.cos(np.radians(dipole_axis_info["dec_deg"]))*np.sin(np.radians(dipole_axis_info["ra_deg"])),
                           np.sin(np.radians(dipole_axis_info["dec_deg"]))])
    compare_directions([vec_bulk, vec_dipole], ["Bulk Flow", "Dipole (z≤0.03)"])

    # Alignment summary
    align_rows = compute_alignment_summary(results_all)
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, ["type","sample_1","sample_2","angle_deg","quality_1","quality_2"])
        w.writeheader(); w.writerows(align_rows)
    print("\n=== Alignment Summary ===")
    for r in align_rows:
        print(f"  {r['type']:24s} | {r['sample_1']:15s} | {r['sample_2']:15s} | angle={r['angle_deg']:8.3f}°")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(make_json_serializable(results_all), f, indent=2)
    print(f"\nAll outputs saved. Total runtime: {time.time()-t0:.1f} s")

if __name__ == "__main__":
    main()
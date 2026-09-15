# # Fermi-LAT Extension Analysis of HESS J1837-069
from pathlib import Path
from fermipy.gtanalysis import GTAnalysis

# ============================================================
# Project configuration
# ============================================================
BASE = Path("/project/ls-gruen/users/asu.uenver/fermi")
CONFIG_FILE = BASE / "hess1837069.yaml"
TARGET_CATALOG_NAME = "FGES J1836.5-0651"


def main():
    print("=" * 60)
    print(f"MULTI-MODEL EXTENSION ANALYSIS — {TARGET_CATALOG_NAME}")
    print("=" * 60)

    gta = GTAnalysis(str(CONFIG_FILE), logging={"verbosity": 3})
    models_to_test = ['RadialGaussian', 'RadialDisk']
    results = {}

    for model in models_to_test:
        print(f"\n>>> Loading optimized ROI state for {model} test...")
        gta.load_roi("hess1837_initial_fit")

        # 1. Freeze all sources, then free only the target source
        gta.free_sources(free=False)
        gta.free_source(TARGET_CATALOG_NAME)
        
        # 2. Re-initialize the internal objects and optimize before extending
        print("Running setup and optimize...")
        gta.setup()
        gta.optimize()

        # 3. Run the extension analysis
        print(f"Running extension analysis using {model}...")
        ext_results = gta.extension(
            TARGET_CATALOG_NAME,
            spatial_model=model,
            width_min=0.01,
            width_max=1.0,
            make_plots=True,
            make_tsmap=True,
            update=True
        )
        
        results[model] = ext_results
        
        # 4. Save the updated state for this specific geometry
        gta.write_roi(f"hess1837_ext_{model.lower()}", make_plots=False)


    print("\n" + "=" * 60)
    print("=== Extension Analysis Comparison ===")
    print("=" * 60)
    
    for model, res in results.items():
        print(f"\nModel: {model}")

        print(f"  Extension TS (Significance): {res['ts_ext']:.2f}") 
        print(f"  Best-fit radius:             {res['ext']:.3f} deg")
        print(f"  Upper limit (95%):           {res['ext_ul95']:.3f} deg")
        
    best_model = max(results, key=lambda k: results[k]['ts_ext'])
    print("\n" + "-" * 60)
    print(f"BEST FIT GEOMETRY: {best_model}")
    print("-" * 60)

if __name__ == "__main__":
    main()
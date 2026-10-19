import os
import argparse

from nnunetv2.preprocessing.preprocessors.dtm_preprocessor import DefaultPreprocessorDTM


def run_preprocess_dtm_entry():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset with DTM (Distance Transform Map) computation. "
                    "Requires ANTsPy. Run AFTER standard nnUNetv2_plan_and_preprocess."
    )
    parser.add_argument("dataset_name_or_id", type=str,
                        help="Dataset name or ID to preprocess with DTM")
    parser.add_argument("configuration", type=str,
                        help="Configuration name (e.g., '3d_fullres', '2d')")
    parser.add_argument("-p", "--plans_identifier", type=str, default="nnUNetPlans",
                        help="Plans identifier (default: nnUNetPlans)")
    parser.add_argument("-np", type=int, default=min(8, os.cpu_count()),
                        help="Number of processes for multiprocessing (default: min(8, cpu_count))")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose output")

    args = parser.parse_args()

    pp = DefaultPreprocessorDTM(verbose=not args.verbose)
    pp.run(args.dataset_name_or_id, args.configuration, args.plans_identifier, args.np)

    print("DTM preprocessing complete.")


if __name__ == "__main__":
    run_preprocess_dtm_entry()

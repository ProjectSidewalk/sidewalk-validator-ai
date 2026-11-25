# Gate block (report-only)

Per run: the val-fitted, uncapped map under the **majority2** deployment prior (the margin prior beside it), applied at the 0.92 gate. gated n, coverage and precision per verdict; Wilson lower bound at z = 1.96 (`retrain_report.WILSON_Z`, as the round-3 report); the confidence where each map first clears the gate; the share of test verdicts at exactly 1.0 (a new head's float saturation moves coverage without any change in discrimination); on held-out, the Agree precision at the gate under the deployments' own class mix. It never changes a verdict.

## NoCurbRamp

| run | prior | split | verdict | gated n | coverage | precision | Wilson lo | clears at | at 1.0 | held-out own-mix Agree precision |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|
| baseline-rerun | majority2 | test | Agree | 998 | 0.746 | 0.983 | 0.973 | 0.999996 | 0.281 |  |
| baseline-rerun | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.430 |  |
| baseline-rerun | majority2 | test_heldout | Agree | 281 | 0.750 | 0.966 | 0.938 | 0.999996 | 0.281 | 0.935 |
| baseline-rerun | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.430 |  |
| baseline-rerun | margin | test | Agree | 387 | 0.284 | 0.965 | 0.942 | 1.0 | 0.281 |  |
| baseline-rerun | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.430 |  |
| baseline-rerun | margin | test_heldout | Agree | 131 | 0.349 | 0.978 | 0.937 | 1.0 | 0.281 | 0.979 |
| baseline-rerun | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.430 |  |
| baseline-rerun-s43 | majority2 | test | Agree | 1019 | 0.765 | 0.986 | 0.977 | 0.99994 | 0.037 |  |
| baseline-rerun-s43 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.253 |  |
| baseline-rerun-s43 | majority2 | test_heldout | Agree | 332 | 0.828 | 0.965 | 0.940 | 0.99994 | 0.037 | 0.933 |
| baseline-rerun-s43 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.253 |  |
| baseline-rerun-s43 | margin | test | Agree | 851 | 0.627 | 0.973 | 0.960 | 0.99998 | 0.037 |  |
| baseline-rerun-s43 | margin | test | Disagree | 313 | 0.243 | 0.961 | 0.934 | 1.0 | 0.253 |  |
| baseline-rerun-s43 | margin | test_heldout | Agree | 300 | 0.720 | 0.936 | 0.902 | 0.99998 | 0.037 | 0.937 |
| baseline-rerun-s43 | margin | test_heldout | Disagree | 47 | 0.151 | 0.962 | 0.865 | 1.0 | 0.253 |  |
| baseline-rerun-s44 | majority2 | test | Agree | 944 | 0.698 | 0.985 | 0.975 | 0.999989 | 0.137 |  |
| baseline-rerun-s44 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.108 |  |
| baseline-rerun-s44 | majority2 | test_heldout | Agree | 263 | 0.673 | 0.969 | 0.940 | 0.999989 | 0.137 | 0.940 |
| baseline-rerun-s44 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.108 |  |
| baseline-rerun-s44 | margin | test | Agree | 944 | 0.681 | 0.970 | 0.957 | 0.999989 | 0.137 |  |
| baseline-rerun-s44 | margin | test | Disagree | 130 | 0.103 | 0.979 | 0.938 | 1.0 | 0.108 |  |
| baseline-rerun-s44 | margin | test_heldout | Agree | 263 | 0.641 | 0.938 | 0.902 | 0.999989 | 0.137 | 0.940 |
| baseline-rerun-s44 | margin | test_heldout | Disagree | 8 | 0.025 | 1.000 | 0.676 | 1.0 | 0.108 |  |
| baseline-rerun-s45 | majority2 | test | Agree | 484 | 0.368 | 0.989 | 0.976 | 1.0 | 0.353 |  |
| baseline-rerun-s45 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.227 |  |
| baseline-rerun-s45 | majority2 | test_heldout | Agree | 165 | 0.452 | 0.981 | 0.947 | 1.0 | 0.353 | 0.963 |
| baseline-rerun-s45 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.227 |  |
| baseline-rerun-s45 | margin | test | Agree | 484 | 0.359 | 0.979 | 0.961 | 1.0 | 0.353 |  |
| baseline-rerun-s45 | margin | test | Disagree | 281 | 0.217 | 0.948 | 0.915 | 1.0 | 0.227 |  |
| baseline-rerun-s45 | margin | test_heldout | Agree | 165 | 0.429 | 0.962 | 0.920 | 1.0 | 0.353 | 0.963 |
| baseline-rerun-s45 | margin | test_heldout | Disagree | 29 | 0.083 | 0.940 | 0.792 | 1.0 | 0.227 |  |
| baseline-rerun-s46 | majority2 | test | Agree | 931 | 0.675 | 0.983 | 0.972 | 0.999999 | 0.385 |  |
| baseline-rerun-s46 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.224 |  |
| baseline-rerun-s46 | majority2 | test_heldout | Agree | 274 | 0.678 | 0.964 | 0.935 | 0.999999 | 0.385 | 0.931 |
| baseline-rerun-s46 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.224 |  |
| baseline-rerun-s46 | margin | test | Agree | 555 | 0.393 | 0.972 | 0.955 | 1.0 | 0.385 |  |
| baseline-rerun-s46 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.224 |  |
| baseline-rerun-s46 | margin | test_heldout | Agree | 174 | 0.417 | 0.947 | 0.902 | 1.0 | 0.385 | 0.948 |
| baseline-rerun-s46 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.224 |  |
| baseline | majority2 | test | Agree | 984 | 0.708 | 0.983 | 0.973 | 0.9993 | 0.006 |  |
| baseline | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.011 |  |
| baseline | majority2 | test_heldout | Agree | 260 | 0.630 | 0.950 | 0.917 | 0.9993 | 0.006 | 0.906 |
| baseline | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.011 |  |
| baseline | margin | test | Agree | 627 | 0.440 | 0.973 | 0.957 | 0.9999 | 0.006 |  |
| baseline | margin | test | Disagree | 249 | 0.206 | 0.978 | 0.952 | 0.999994 | 0.011 |  |
| baseline | margin | test_heldout | Agree | 158 | 0.372 | 0.933 | 0.883 | 0.9999 | 0.006 | 0.935 |
| baseline | margin | test_heldout | Disagree | 17 | 0.056 | 1.000 | 0.816 | 0.999994 | 0.011 |  |
| baseline-s43 | majority2 | test | Agree | 1073 | 0.783 | 0.984 | 0.975 | 0.999996 | 0.241 |  |
| baseline-s43 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.314 |  |
| baseline-s43 | majority2 | test_heldout | Agree | 314 | 0.763 | 0.960 | 0.932 | 0.999996 | 0.241 | 0.923 |
| baseline-s43 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.314 |  |
| baseline-s43 | margin | test | Agree | 346 | 0.247 | 0.979 | 0.957 | 1.0 | 0.241 |  |
| baseline-s43 | margin | test | Disagree | 370 | 0.305 | 0.924 | 0.893 | 1.0 | 0.314 |  |
| baseline-s43 | margin | test_heldout | Agree | 110 | 0.262 | 0.953 | 0.895 | 1.0 | 0.241 | 0.954 |
| baseline-s43 | margin | test_heldout | Disagree | 53 | 0.175 | 1.000 | 0.932 | 1.0 | 0.314 |  |
| baseline-s44 | majority2 | test | Agree | 1004 | 0.763 | 0.985 | 0.975 | 0.99998 | 0.113 |  |
| baseline-s44 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.318 |  |
| baseline-s44 | majority2 | test_heldout | Agree | 296 | 0.792 | 0.972 | 0.947 | 0.99998 | 0.113 | 0.947 |
| baseline-s44 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.318 |  |
| baseline-s44 | margin | test | Agree | 737 | 0.551 | 0.973 | 0.958 | 0.999998 | 0.113 |  |
| baseline-s44 | margin | test | Disagree | 399 | 0.305 | 0.936 | 0.908 | 1.0 | 0.318 |  |
| baseline-s44 | margin | test_heldout | Agree | 230 | 0.599 | 0.963 | 0.929 | 0.999998 | 0.113 | 0.964 |
| baseline-s44 | margin | test_heldout | Disagree | 47 | 0.138 | 0.891 | 0.771 | 1.0 | 0.318 |  |
| two_scale | majority2 | test | Agree | 888 | 0.670 | 0.988 | 0.978 | 0.999999 | 0.491 |  |
| two_scale | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.349 |  |
| two_scale | majority2 | test_heldout | Agree | 225 | 0.598 | 0.978 | 0.950 | 0.999999 | 0.491 | 0.958 |
| two_scale | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.349 |  |
| two_scale | margin | test | Agree | 888 | 0.651 | 0.975 | 0.962 | 0.999999 | 0.491 |  |
| two_scale | margin | test | Disagree | 427 | 0.335 | 0.937 | 0.910 | 1.0 | 0.349 |  |
| two_scale | margin | test_heldout | Agree | 225 | 0.568 | 0.956 | 0.921 | 0.999999 | 0.491 | 0.958 |
| two_scale | margin | test_heldout | Disagree | 24 | 0.075 | 0.860 | 0.672 | 1.0 | 0.349 |  |
| two_scale-s43 | majority2 | test | Agree | 894 | 0.681 | 0.989 | 0.980 | 1.0 | 0.655 |  |
| two_scale-s43 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.484 |  |
| two_scale-s43 | majority2 | test_heldout | Agree | 221 | 0.574 | 0.974 | 0.943 | 1.0 | 0.655 | 0.949 |
| two_scale-s43 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.484 |  |
| two_scale-s43 | margin | test | Agree | 894 | 0.666 | 0.978 | 0.967 | 1.0 | 0.655 |  |
| two_scale-s43 | margin | test | Disagree | 604 | 0.465 | 0.942 | 0.921 | 1.0 | 0.484 |  |
| two_scale-s43 | margin | test_heldout | Agree | 221 | 0.549 | 0.947 | 0.910 | 1.0 | 0.655 | 0.949 |
| two_scale-s43 | margin | test_heldout | Disagree | 83 | 0.248 | 1.000 | 0.956 | 1.0 | 0.484 |  |
| two_scale-s44 | majority2 | test | Agree | 872 | 0.652 | 0.990 | 0.980 | 1.0 | 0.626 |  |
| two_scale-s44 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.519 |  |
| two_scale-s44 | majority2 | test_heldout | Agree | 218 | 0.578 | 0.978 | 0.948 | 1.0 | 0.626 | 0.956 |
| two_scale-s44 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.519 |  |
| two_scale-s44 | margin | test | Agree | 872 | 0.637 | 0.979 | 0.967 | 1.0 | 0.626 |  |
| two_scale-s44 | margin | test | Disagree | 634 | 0.504 | 0.922 | 0.899 | 1.0 | 0.519 |  |
| two_scale-s44 | margin | test_heldout | Agree | 218 | 0.549 | 0.955 | 0.919 | 1.0 | 0.626 | 0.956 |
| two_scale-s44 | margin | test_heldout | Disagree | 81 | 0.243 | 0.957 | 0.888 | 1.0 | 0.519 |  |
| two_scale-s45 | majority2 | test | Agree | 872 | 0.653 | 0.988 | 0.979 | 1.0 | 0.629 |  |
| two_scale-s45 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.553 |  |
| two_scale-s45 | majority2 | test_heldout | Agree | 243 | 0.628 | 0.979 | 0.952 | 1.0 | 0.629 | 0.959 |
| two_scale-s45 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.553 |  |
| two_scale-s45 | margin | test | Agree | 872 | 0.638 | 0.976 | 0.964 | 1.0 | 0.629 |  |
| two_scale-s45 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.553 |  |
| two_scale-s45 | margin | test_heldout | Agree | 243 | 0.600 | 0.957 | 0.924 | 1.0 | 0.629 | 0.959 |
| two_scale-s45 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.553 |  |
| two_scale-s46 | majority2 | test | Agree | 947 | 0.687 | 0.986 | 0.977 | 1.0 | 0.658 |  |
| two_scale-s46 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.435 |  |
| two_scale-s46 | majority2 | test_heldout | Agree | 250 | 0.625 | 0.976 | 0.948 | 1.0 | 0.658 | 0.953 |
| two_scale-s46 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.435 |  |
| two_scale-s46 | margin | test | Agree | 947 | 0.670 | 0.973 | 0.960 | 1.0 | 0.658 |  |
| two_scale-s46 | margin | test | Disagree | 511 | 0.422 | 0.945 | 0.922 | 1.0 | 0.435 |  |
| two_scale-s46 | margin | test_heldout | Agree | 250 | 0.595 | 0.951 | 0.917 | 1.0 | 0.658 | 0.953 |
| two_scale-s46 | margin | test_heldout | Disagree | 60 | 0.201 | 0.971 | 0.892 | 1.0 | 0.435 |  |
| view_heatmap | majority2 | test | Agree | 525 | 0.392 | 0.985 | 0.971 | 1.0 | 0.362 |  |
| view_heatmap | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.248 |  |
| view_heatmap | majority2 | test_heldout | Agree | 130 | 0.350 | 0.961 | 0.912 | 1.0 | 0.362 | 0.925 |
| view_heatmap | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.248 |  |
| view_heatmap | margin | test | Agree | 525 | 0.373 | 0.970 | 0.952 | 1.0 | 0.362 |  |
| view_heatmap | margin | test | Disagree | 288 | 0.237 | 0.921 | 0.885 | 1.0 | 0.248 |  |
| view_heatmap | margin | test_heldout | Agree | 130 | 0.328 | 0.923 | 0.864 | 1.0 | 0.362 | 0.925 |
| view_heatmap | margin | test_heldout | Disagree | 26 | 0.075 | 0.933 | 0.772 | 1.0 | 0.248 |  |
| view_heatmap-s43 | majority2 | test | Agree | 437 | 0.331 | 0.981 | 0.964 | 1.0 | 0.309 |  |
| view_heatmap-s43 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.343 |  |
| view_heatmap-s43 | majority2 | test_heldout | Agree | 95 | 0.256 | 0.959 | 0.898 | 1.0 | 0.309 | 0.921 |
| view_heatmap-s43 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.343 |  |
| view_heatmap-s43 | margin | test | Agree | 437 | 0.318 | 0.962 | 0.940 | 1.0 | 0.309 |  |
| view_heatmap-s43 | margin | test | Disagree | 412 | 0.327 | 0.916 | 0.885 | 1.0 | 0.343 |  |
| view_heatmap-s43 | margin | test_heldout | Agree | 95 | 0.239 | 0.919 | 0.846 | 1.0 | 0.309 | 0.921 |
| view_heatmap-s43 | margin | test_heldout | Disagree | 44 | 0.127 | 0.921 | 0.804 | 1.0 | 0.343 |  |
| view_heatmap-s44 | majority2 | test | Agree | 836 | 0.639 | 0.981 | 0.969 | 0.999999 | 0.390 |  |
| view_heatmap-s44 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.132 |  |
| view_heatmap-s44 | majority2 | test_heldout | Agree | 194 | 0.532 | 0.958 | 0.920 | 0.999999 | 0.390 | 0.920 |
| view_heatmap-s44 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.132 |  |
| view_heatmap-s44 | margin | test | Agree | 836 | 0.616 | 0.962 | 0.946 | 0.999999 | 0.390 |  |
| view_heatmap-s44 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.132 |  |
| view_heatmap-s44 | margin | test_heldout | Agree | 194 | 0.505 | 0.917 | 0.870 | 0.999999 | 0.390 | 0.920 |
| view_heatmap-s44 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.132 |  |
| view_heatmap-s45 | majority2 | test | Agree | 878 | 0.650 | 0.979 | 0.967 | 0.999999 | 0.422 |  |
| view_heatmap-s45 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.361 |  |
| view_heatmap-s45 | majority2 | test_heldout | Agree | 214 | 0.546 | 0.945 | 0.905 | 0.999999 | 0.422 | 0.896 |
| view_heatmap-s45 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.361 |  |
| view_heatmap-s45 | margin | test | Agree | 608 | 0.436 | 0.976 | 0.960 | 1.0 | 0.422 |  |
| view_heatmap-s45 | margin | test | Disagree | 423 | 0.345 | 0.924 | 0.895 | 1.0 | 0.361 |  |
| view_heatmap-s45 | margin | test_heldout | Agree | 133 | 0.331 | 0.924 | 0.867 | 1.0 | 0.422 | 0.927 |
| view_heatmap-s45 | margin | test_heldout | Disagree | 43 | 0.126 | 0.959 | 0.853 | 1.0 | 0.361 |  |
| view_heatmap-s46 | majority2 | test | Agree | 948 | 0.711 | 0.974 | 0.962 | 0.99998 | 0.218 |  |
| view_heatmap-s46 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.228 |  |
| view_heatmap-s46 | majority2 | test_heldout | Agree | 234 | 0.620 | 0.942 | 0.905 | 0.99998 | 0.218 | 0.892 |
| view_heatmap-s46 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.228 |  |
| view_heatmap-s46 | margin | test | Agree | 306 | 0.224 | 0.971 | 0.945 | 1.0 | 0.218 |  |
| view_heatmap-s46 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.228 |  |
| view_heatmap-s46 | margin | test_heldout | Agree | 96 | 0.253 | 0.945 | 0.881 | 1.0 | 0.218 | 0.947 |
| view_heatmap-s46 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.228 |  |
| view_heatmap-r448 | majority2 | test | Agree | 597 | 0.434 | 0.988 | 0.976 | 1.0 | 0.398 |  |
| view_heatmap-r448 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.235 |  |
| view_heatmap-r448 | majority2 | test_heldout | Agree | 144 | 0.377 | 0.980 | 0.942 | 1.0 | 0.398 | 0.961 |
| view_heatmap-r448 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.235 |  |
| view_heatmap-r448 | margin | test | Agree | 597 | 0.412 | 0.976 | 0.961 | 1.0 | 0.398 |  |
| view_heatmap-r448 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.235 |  |
| view_heatmap-r448 | margin | test_heldout | Agree | 144 | 0.349 | 0.960 | 0.914 | 1.0 | 0.398 | 0.961 |
| view_heatmap-r448 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.235 |  |
| side_info | majority2 | test | Agree | 495 | 0.376 | 0.988 | 0.974 | 1.0 | 0.362 |  |
| side_info | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.242 |  |
| side_info | majority2 | test_heldout | Agree | 155 | 0.414 | 0.976 | 0.938 | 1.0 | 0.362 | 0.953 |
| side_info | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.242 |  |
| side_info | margin | test | Agree | 495 | 0.367 | 0.976 | 0.958 | 1.0 | 0.362 |  |
| side_info | margin | test | Disagree | 301 | 0.233 | 0.933 | 0.899 | 1.0 | 0.242 |  |
| side_info | margin | test_heldout | Agree | 155 | 0.393 | 0.951 | 0.905 | 1.0 | 0.362 | 0.953 |
| side_info | margin | test_heldout | Disagree | 33 | 0.095 | 1.000 | 0.896 | 1.0 | 0.242 |  |
| side_info_nohq | majority2 | test | Agree | 873 | 0.641 | 0.985 | 0.975 | 0.99998 | 0.048 |  |
| side_info_nohq | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.267 |  |
| side_info_nohq | majority2 | test_heldout | Agree | 275 | 0.684 | 0.966 | 0.937 | 0.99998 | 0.048 | 0.934 |
| side_info_nohq | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.267 |  |
| side_info_nohq | margin | test | Agree | 658 | 0.472 | 0.973 | 0.957 | 0.999994 | 0.048 |  |
| side_info_nohq | margin | test | Disagree | 318 | 0.259 | 0.925 | 0.890 | 1.0 | 0.267 |  |
| side_info_nohq | margin | test_heldout | Agree | 214 | 0.515 | 0.951 | 0.914 | 0.999994 | 0.048 | 0.953 |
| side_info_nohq | margin | test_heldout | Disagree | 21 | 0.070 | 0.918 | 0.727 | 1.0 | 0.267 |  |

## Obstacle

| run | prior | split | verdict | gated n | coverage | precision | Wilson lo | clears at | at 1.0 | held-out own-mix Agree precision |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|
| baseline-rerun | majority2 | test | Agree | 574 | 0.481 | 0.983 | 0.968 | 1.0 | 0.451 |  |
| baseline-rerun | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.339 |  |
| baseline-rerun | majority2 | test_heldout | Agree | 304 | 0.521 | 0.986 | 0.965 | 1.0 | 0.451 | 0.946 |
| baseline-rerun | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.339 |  |
| baseline-rerun | margin | test | Agree | 574 | 0.464 | 0.972 | 0.955 | 1.0 | 0.451 |  |
| baseline-rerun | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.339 |  |
| baseline-rerun | margin | test_heldout | Agree | 304 | 0.491 | 0.977 | 0.953 | 1.0 | 0.451 | 0.946 |
| baseline-rerun | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.339 |  |
| baseline-rerun-s43 | majority2 | test | Agree | 429 | 0.364 | 0.980 | 0.962 | 1.0 | 0.341 |  |
| baseline-rerun-s43 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.305 |  |
| baseline-rerun-s43 | majority2 | test_heldout | Agree | 241 | 0.444 | 0.993 | 0.972 | 1.0 | 0.341 | 0.973 |
| baseline-rerun-s43 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.305 |  |
| baseline-rerun-s43 | margin | test | Agree | 429 | 0.351 | 0.968 | 0.947 | 1.0 | 0.341 |  |
| baseline-rerun-s43 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.305 |  |
| baseline-rerun-s43 | margin | test_heldout | Agree | 241 | 0.422 | 0.989 | 0.966 | 1.0 | 0.341 | 0.973 |
| baseline-rerun-s43 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.305 |  |
| baseline-rerun-s44 | majority2 | test | Agree | 669 | 0.517 | 0.972 | 0.957 | 1.0 | 0.480 |  |
| baseline-rerun-s44 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.285 |  |
| baseline-rerun-s44 | majority2 | test_heldout | Agree | 350 | 0.560 | 0.982 | 0.961 | 1.0 | 0.480 | 0.932 |
| baseline-rerun-s44 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.285 |  |
| baseline-rerun-s44 | margin | test | Agree | 669 | 0.496 | 0.955 | 0.937 | 1.0 | 0.480 |  |
| baseline-rerun-s44 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.285 |  |
| baseline-rerun-s44 | margin | test_heldout | Agree | 350 | 0.527 | 0.970 | 0.947 | 1.0 | 0.480 | 0.932 |
| baseline-rerun-s44 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.285 |  |
| baseline-rerun-s45 | majority2 | test | Agree | 672 | 0.540 | 0.975 | 0.960 | 1.0 | 0.505 |  |
| baseline-rerun-s45 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.303 |  |
| baseline-rerun-s45 | majority2 | test_heldout | Agree | 342 | 0.571 | 0.986 | 0.967 | 1.0 | 0.505 | 0.946 |
| baseline-rerun-s45 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.303 |  |
| baseline-rerun-s45 | margin | test | Agree | 672 | 0.520 | 0.959 | 0.941 | 1.0 | 0.505 |  |
| baseline-rerun-s45 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.303 |  |
| baseline-rerun-s45 | margin | test_heldout | Agree | 342 | 0.534 | 0.977 | 0.955 | 1.0 | 0.505 | 0.946 |
| baseline-rerun-s45 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.303 |  |
| baseline-rerun-s46 | majority2 | test | Agree | 540 | 0.448 | 0.976 | 0.960 | 1.0 | 0.419 |  |
| baseline-rerun-s46 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.282 |  |
| baseline-rerun-s46 | majority2 | test_heldout | Agree | 260 | 0.465 | 0.991 | 0.970 | 1.0 | 0.419 | 0.965 |
| baseline-rerun-s46 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.282 |  |
| baseline-rerun-s46 | margin | test | Agree | 540 | 0.431 | 0.962 | 0.942 | 1.0 | 0.419 |  |
| baseline-rerun-s46 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.282 |  |
| baseline-rerun-s46 | margin | test_heldout | Agree | 260 | 0.439 | 0.985 | 0.962 | 1.0 | 0.419 | 0.965 |
| baseline-rerun-s46 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.282 |  |
| baseline | majority2 | test | Agree | 631 | 0.494 | 0.974 | 0.959 | 1.0 | 0.461 |  |
| baseline | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.344 |  |
| baseline | majority2 | test_heldout | Agree | 325 | 0.520 | 0.981 | 0.960 | 1.0 | 0.461 | 0.931 |
| baseline | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.344 |  |
| baseline | margin | test | Agree | 631 | 0.475 | 0.959 | 0.940 | 1.0 | 0.461 |  |
| baseline | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.344 |  |
| baseline | margin | test_heldout | Agree | 325 | 0.489 | 0.970 | 0.945 | 1.0 | 0.461 | 0.931 |
| baseline | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.344 |  |
| baseline-s43 | majority2 | test | Agree | 512 | 0.425 | 0.979 | 0.962 | 1.0 | 0.397 |  |
| baseline-s43 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.319 |  |
| baseline-s43 | majority2 | test_heldout | Agree | 268 | 0.458 | 0.986 | 0.964 | 1.0 | 0.397 | 0.949 |
| baseline-s43 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.319 |  |
| baseline-s43 | margin | test | Agree | 512 | 0.409 | 0.966 | 0.946 | 1.0 | 0.397 |  |
| baseline-s43 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.319 |  |
| baseline-s43 | margin | test_heldout | Agree | 268 | 0.433 | 0.978 | 0.952 | 1.0 | 0.397 | 0.949 |
| baseline-s43 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.319 |  |
| baseline-s44 | majority2 | test | Agree | 603 | 0.493 | 0.977 | 0.962 | 1.0 | 0.461 |  |
| baseline-s44 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.382 |  |
| baseline-s44 | majority2 | test_heldout | Agree | 297 | 0.516 | 0.988 | 0.968 | 1.0 | 0.461 | 0.954 |
| baseline-s44 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.382 |  |
| baseline-s44 | margin | test | Agree | 603 | 0.475 | 0.963 | 0.945 | 1.0 | 0.461 |  |
| baseline-s44 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.382 |  |
| baseline-s44 | margin | test_heldout | Agree | 297 | 0.490 | 0.980 | 0.957 | 1.0 | 0.461 | 0.954 |
| baseline-s44 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.382 |  |
| two_scale | majority2 | test | Agree | 492 | 0.398 | 0.977 | 0.959 | 1.0 | 0.370 |  |
| two_scale | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.318 |  |
| two_scale | majority2 | test_heldout | Agree | 256 | 0.449 | 0.991 | 0.970 | 1.0 | 0.370 | 0.965 |
| two_scale | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.318 |  |
| two_scale | margin | test | Agree | 492 | 0.382 | 0.963 | 0.942 | 1.0 | 0.370 |  |
| two_scale | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.318 |  |
| two_scale | margin | test_heldout | Agree | 256 | 0.423 | 0.985 | 0.961 | 1.0 | 0.370 | 0.965 |
| two_scale | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.318 |  |
| two_scale-s43 | majority2 | test | Agree | 476 | 0.397 | 0.985 | 0.970 | 1.0 | 0.372 |  |
| two_scale-s43 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.290 |  |
| two_scale-s43 | majority2 | test_heldout | Agree | 272 | 0.464 | 0.991 | 0.971 | 1.0 | 0.372 | 0.967 |
| two_scale-s43 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.290 |  |
| two_scale-s43 | margin | test | Agree | 476 | 0.383 | 0.976 | 0.958 | 1.0 | 0.372 |  |
| two_scale-s43 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.290 |  |
| two_scale-s43 | margin | test_heldout | Agree | 272 | 0.436 | 0.986 | 0.963 | 1.0 | 0.372 | 0.967 |
| two_scale-s43 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.290 |  |
| two_scale-s44 | majority2 | test | Agree | 475 | 0.385 | 0.982 | 0.966 | 1.0 | 0.356 |  |
| two_scale-s44 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.300 |  |
| two_scale-s44 | majority2 | test_heldout | Agree | 247 | 0.394 | 0.983 | 0.958 | 1.0 | 0.356 | 0.936 |
| two_scale-s44 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.300 |  |
| two_scale-s44 | margin | test | Agree | 475 | 0.369 | 0.971 | 0.952 | 1.0 | 0.356 |  |
| two_scale-s44 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.300 |  |
| two_scale-s44 | margin | test_heldout | Agree | 247 | 0.367 | 0.972 | 0.943 | 1.0 | 0.356 | 0.936 |
| two_scale-s44 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.300 |  |
| two_scale-s45 | majority2 | test | Agree | 560 | 0.450 | 0.982 | 0.968 | 1.0 | 0.415 |  |
| two_scale-s45 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.276 |  |
| two_scale-s45 | majority2 | test_heldout | Agree | 297 | 0.470 | 0.985 | 0.964 | 1.0 | 0.415 | 0.945 |
| two_scale-s45 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.276 |  |
| two_scale-s45 | margin | test | Agree | 560 | 0.430 | 0.971 | 0.954 | 1.0 | 0.415 |  |
| two_scale-s45 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.276 |  |
| two_scale-s45 | margin | test_heldout | Agree | 297 | 0.437 | 0.976 | 0.952 | 1.0 | 0.415 | 0.945 |
| two_scale-s45 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.276 |  |
| two_scale-s46 | majority2 | test | Agree | 513 | 0.430 | 0.983 | 0.968 | 1.0 | 0.403 |  |
| two_scale-s46 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.300 |  |
| two_scale-s46 | majority2 | test_heldout | Agree | 279 | 0.449 | 0.981 | 0.958 | 1.0 | 0.403 | 0.931 |
| two_scale-s46 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.300 |  |
| two_scale-s46 | margin | test | Agree | 513 | 0.415 | 0.973 | 0.955 | 1.0 | 0.403 |  |
| two_scale-s46 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.300 |  |
| two_scale-s46 | margin | test_heldout | Agree | 279 | 0.420 | 0.970 | 0.943 | 1.0 | 0.403 | 0.931 |
| two_scale-s46 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.300 |  |
| view_heatmap | majority2 | test | Agree | 443 | 0.364 | 0.966 | 0.945 | 0.999998 | 0.113 |  |
| view_heatmap | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.170 |  |
| view_heatmap | majority2 | test_heldout | Agree | 237 | 0.383 | 0.974 | 0.945 | 0.999998 | 0.113 | 0.907 |
| view_heatmap | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.170 |  |
| view_heatmap | margin | test | Agree | 148 | 0.117 | 0.964 | 0.920 | 1.0 | 0.113 |  |
| view_heatmap | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.170 |  |
| view_heatmap | margin | test_heldout | Agree | 67 | 0.099 | 0.950 | 0.870 | 1.0 | 0.113 | 0.889 |
| view_heatmap | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.170 |  |
| view_heatmap-s43 | majority2 | test | Agree | 199 | 0.161 | 0.979 | 0.948 | 1.0 | 0.146 |  |
| view_heatmap-s43 | majority2 | test | Disagree | 128 | 0.119 | 0.845 | 0.772 | 1.0 | 0.142 |  |
| view_heatmap-s43 | majority2 | test_heldout | Agree | 91 | 0.149 | 0.980 | 0.927 | 1.0 | 0.146 | 0.927 |
| view_heatmap-s43 | majority2 | test_heldout | Disagree | 23 | 0.020 | 0.782 | 0.580 | 1.0 | 0.142 |  |
| view_heatmap-s43 | margin | test | Agree | 199 | 0.152 | 0.965 | 0.930 | 1.0 | 0.146 |  |
| view_heatmap-s43 | margin | test | Disagree | 128 | 0.134 | 0.899 | 0.835 | 1.0 | 0.142 |  |
| view_heatmap-s43 | margin | test_heldout | Agree | 91 | 0.139 | 0.968 | 0.909 | 1.0 | 0.146 | 0.927 |
| view_heatmap-s43 | margin | test_heldout | Disagree | 23 | 0.022 | 0.854 | 0.661 | 1.0 | 0.142 |  |
| view_heatmap-s44 | majority2 | test | Agree | 352 | 0.294 | 0.972 | 0.948 | 1.0 | 0.273 |  |
| view_heatmap-s44 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.239 |  |
| view_heatmap-s44 | majority2 | test_heldout | Agree | 229 | 0.382 | 0.984 | 0.958 | 1.0 | 0.273 | 0.940 |
| view_heatmap-s44 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.239 |  |
| view_heatmap-s44 | margin | test | Agree | 352 | 0.282 | 0.954 | 0.927 | 1.0 | 0.273 |  |
| view_heatmap-s44 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.239 |  |
| view_heatmap-s44 | margin | test_heldout | Agree | 229 | 0.352 | 0.974 | 0.944 | 1.0 | 0.273 | 0.940 |
| view_heatmap-s44 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.239 |  |
| view_heatmap-s45 | majority2 | test | Agree | 416 | 0.343 | 0.976 | 0.956 | 1.0 | 0.313 |  |
| view_heatmap-s45 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.225 |  |
| view_heatmap-s45 | majority2 | test_heldout | Agree | 251 | 0.432 | 0.986 | 0.963 | 1.0 | 0.313 | 0.948 |
| view_heatmap-s45 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.225 |  |
| view_heatmap-s45 | margin | test | Agree | 416 | 0.325 | 0.961 | 0.938 | 1.0 | 0.313 |  |
| view_heatmap-s45 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.225 |  |
| view_heatmap-s45 | margin | test_heldout | Agree | 251 | 0.404 | 0.977 | 0.951 | 1.0 | 0.313 | 0.948 |
| view_heatmap-s45 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.225 |  |
| view_heatmap-s46 | majority2 | test | Agree | 656 | 0.516 | 0.953 | 0.934 | 0.999995 | 0.226 |  |
| view_heatmap-s46 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.090 |  |
| view_heatmap-s46 | majority2 | test_heldout | Agree | 372 | 0.576 | 0.972 | 0.950 | 0.999995 | 0.226 | 0.899 |
| view_heatmap-s46 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.090 |  |
| view_heatmap-s46 | margin | test | Agree | 309 | 0.235 | 0.953 | 0.923 | 1.0 | 0.226 |  |
| view_heatmap-s46 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.090 |  |
| view_heatmap-s46 | margin | test_heldout | Agree | 207 | 0.318 | 0.980 | 0.950 | 1.0 | 0.226 | 0.953 |
| view_heatmap-s46 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.090 |  |
| view_heatmap-r448 | majority2 | test | Agree | 530 | 0.428 | 0.979 | 0.964 | 1.0 | 0.396 |  |
| view_heatmap-r448 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.208 |  |
| view_heatmap-r448 | majority2 | test_heldout | Agree | 277 | 0.486 | 0.989 | 0.969 | 1.0 | 0.396 | 0.960 |
| view_heatmap-r448 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.208 |  |
| view_heatmap-r448 | margin | test | Agree | 530 | 0.410 | 0.967 | 0.948 | 1.0 | 0.396 |  |
| view_heatmap-r448 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.208 |  |
| view_heatmap-r448 | margin | test_heldout | Agree | 277 | 0.458 | 0.983 | 0.960 | 1.0 | 0.396 | 0.960 |
| view_heatmap-r448 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.208 |  |
| side_info | majority2 | test | Agree | 452 | 0.366 | 0.981 | 0.964 | 1.0 | 0.338 |  |
| side_info | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.225 |  |
| side_info | majority2 | test_heldout | Agree | 246 | 0.410 | 0.991 | 0.969 | 1.0 | 0.338 | 0.966 |
| side_info | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.225 |  |
| side_info | margin | test | Agree | 452 | 0.350 | 0.970 | 0.949 | 1.0 | 0.338 |  |
| side_info | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.225 |  |
| side_info | margin | test_heldout | Agree | 246 | 0.380 | 0.985 | 0.961 | 1.0 | 0.338 | 0.966 |
| side_info | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.225 |  |
| side_info_nohq | majority2 | test | Agree | 410 | 0.348 | 0.983 | 0.965 | 0.95 | 0.000 |  |
| side_info_nohq | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.000 |  |
| side_info_nohq | majority2 | test_heldout | Agree | 220 | 0.389 | 0.991 | 0.968 | 0.95 | 0.000 | 0.968 |
| side_info_nohq | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.000 |  |
| side_info_nohq | margin | test | Agree | 293 | 0.239 | 0.969 | 0.942 | 0.9675 | 0.000 |  |
| side_info_nohq | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.000 |  |
| side_info_nohq | margin | test_heldout | Agree | 165 | 0.279 | 0.990 | 0.960 | 0.9675 | 0.000 | 0.977 |
| side_info_nohq | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.000 |  |

## SurfaceProblem

| run | prior | split | verdict | gated n | coverage | precision | Wilson lo | clears at | at 1.0 | held-out own-mix Agree precision |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|
| baseline-rerun | majority2 | test | Agree | 727 | 0.614 | 0.983 | 0.970 | 1.0 | 0.565 |  |
| baseline-rerun | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.360 |  |
| baseline-rerun | majority2 | test_heldout | Agree | 148 | 0.444 | 0.953 | 0.906 | 1.0 | 0.565 | 0.896 |
| baseline-rerun | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.360 |  |
| baseline-rerun | margin | test | Agree | 727 | 0.591 | 0.962 | 0.946 | 1.0 | 0.565 |  |
| baseline-rerun | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.360 |  |
| baseline-rerun | margin | test_heldout | Agree | 148 | 0.417 | 0.901 | 0.843 | 1.0 | 0.565 | 0.896 |
| baseline-rerun | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.360 |  |
| baseline-rerun-s43 | majority2 | test | Agree | 613 | 0.520 | 0.985 | 0.972 | 1.0 | 0.479 |  |
| baseline-rerun-s43 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.371 |  |
| baseline-rerun-s43 | majority2 | test_heldout | Agree | 119 | 0.364 | 0.960 | 0.908 | 1.0 | 0.479 | 0.910 |
| baseline-rerun-s43 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.371 |  |
| baseline-rerun-s43 | margin | test | Agree | 613 | 0.500 | 0.967 | 0.950 | 1.0 | 0.479 |  |
| baseline-rerun-s43 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.371 |  |
| baseline-rerun-s43 | margin | test_heldout | Agree | 119 | 0.339 | 0.915 | 0.851 | 1.0 | 0.479 | 0.910 |
| baseline-rerun-s43 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.371 |  |
| baseline-rerun-s44 | majority2 | test | Agree | 681 | 0.583 | 0.985 | 0.973 | 1.0 | 0.534 |  |
| baseline-rerun-s44 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.386 |  |
| baseline-rerun-s44 | majority2 | test_heldout | Agree | 118 | 0.385 | 0.961 | 0.910 | 1.0 | 0.534 | 0.914 |
| baseline-rerun-s44 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.386 |  |
| baseline-rerun-s44 | margin | test | Agree | 681 | 0.560 | 0.967 | 0.951 | 1.0 | 0.534 |  |
| baseline-rerun-s44 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.386 |  |
| baseline-rerun-s44 | margin | test_heldout | Agree | 118 | 0.357 | 0.919 | 0.855 | 1.0 | 0.534 | 0.914 |
| baseline-rerun-s44 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.386 |  |
| baseline-rerun-s45 | majority2 | test | Agree | 617 | 0.546 | 0.984 | 0.970 | 0.999998 | 0.262 |  |
| baseline-rerun-s45 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.239 |  |
| baseline-rerun-s45 | majority2 | test_heldout | Agree | 93 | 0.371 | 0.967 | 0.908 | 0.999998 | 0.262 | 0.925 |
| baseline-rerun-s45 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.239 |  |
| baseline-rerun-s45 | margin | test | Agree | 617 | 0.529 | 0.965 | 0.947 | 0.999998 | 0.262 |  |
| baseline-rerun-s45 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.239 |  |
| baseline-rerun-s45 | margin | test_heldout | Agree | 93 | 0.348 | 0.929 | 0.858 | 0.999998 | 0.262 | 0.925 |
| baseline-rerun-s45 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.239 |  |
| baseline-rerun-s46 | majority2 | test | Agree | 607 | 0.514 | 0.987 | 0.974 | 1.0 | 0.465 |  |
| baseline-rerun-s46 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.238 |  |
| baseline-rerun-s46 | majority2 | test_heldout | Agree | 103 | 0.304 | 0.954 | 0.895 | 1.0 | 0.465 | 0.899 |
| baseline-rerun-s46 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.238 |  |
| baseline-rerun-s46 | margin | test | Agree | 607 | 0.491 | 0.971 | 0.955 | 1.0 | 0.465 |  |
| baseline-rerun-s46 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.238 |  |
| baseline-rerun-s46 | margin | test_heldout | Agree | 103 | 0.283 | 0.905 | 0.833 | 1.0 | 0.465 | 0.899 |
| baseline-rerun-s46 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.238 |  |
| baseline | majority2 | test | Agree | 819 | 0.696 | 0.980 | 0.968 | 0.999997 | 0.293 |  |
| baseline | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.359 |  |
| baseline | majority2 | test_heldout | Agree | 178 | 0.554 | 0.947 | 0.903 | 0.999997 | 0.293 | 0.883 |
| baseline | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.359 |  |
| baseline | margin | test | Agree | 819 | 0.676 | 0.957 | 0.941 | 0.999997 | 0.293 |  |
| baseline | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.359 |  |
| baseline | margin | test_heldout | Agree | 178 | 0.528 | 0.890 | 0.835 | 0.999997 | 0.293 | 0.883 |
| baseline | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.359 |  |
| baseline-s43 | majority2 | test | Agree | 567 | 0.492 | 0.989 | 0.977 | 0.95 | 0.000 |  |
| baseline-s43 | majority2 | test | Disagree | 34 | 0.023 | 1.000 | 0.898 | 0.999 | 0.000 |  |
| baseline-s43 | majority2 | test_heldout | Agree | 112 | 0.432 | 0.975 | 0.927 | 0.95 | 0.000 | 0.943 |
| baseline-s43 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | 0.999 | 0.000 |  |
| baseline-s43 | margin | test | Agree | 567 | 0.474 | 0.977 | 0.961 | 0.95 | 0.000 |  |
| baseline-s43 | margin | test | Disagree | 34 | 0.030 | 1.000 | 0.898 | 0.999 | 0.000 |  |
| baseline-s43 | margin | test_heldout | Agree | 112 | 0.399 | 0.946 | 0.888 | 0.95 | 0.000 | 0.943 |
| baseline-s43 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | 0.999 | 0.000 |  |
| baseline-s44 | majority2 | test | Agree | 740 | 0.622 | 0.981 | 0.969 | 1.0 | 0.573 |  |
| baseline-s44 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.378 |  |
| baseline-s44 | majority2 | test_heldout | Agree | 147 | 0.489 | 0.959 | 0.914 | 1.0 | 0.573 | 0.909 |
| baseline-s44 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.378 |  |
| baseline-s44 | margin | test | Agree | 740 | 0.599 | 0.960 | 0.943 | 1.0 | 0.573 |  |
| baseline-s44 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.378 |  |
| baseline-s44 | margin | test_heldout | Agree | 147 | 0.458 | 0.914 | 0.858 | 1.0 | 0.573 | 0.909 |
| baseline-s44 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.378 |  |
| two_scale | majority2 | test | Agree | 689 | 0.569 | 0.981 | 0.968 | 1.0 | 0.520 |  |
| two_scale | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.301 |  |
| two_scale | majority2 | test_heldout | Agree | 137 | 0.460 | 0.963 | 0.916 | 1.0 | 0.520 | 0.916 |
| two_scale | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.301 |  |
| two_scale | margin | test | Agree | 689 | 0.546 | 0.959 | 0.942 | 1.0 | 0.520 |  |
| two_scale | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.301 |  |
| two_scale | margin | test_heldout | Agree | 137 | 0.430 | 0.921 | 0.863 | 1.0 | 0.520 | 0.916 |
| two_scale | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.301 |  |
| two_scale-s43 | majority2 | test | Agree | 782 | 0.644 | 0.978 | 0.965 | 1.0 | 0.596 |  |
| two_scale-s43 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.332 |  |
| two_scale-s43 | majority2 | test_heldout | Agree | 165 | 0.532 | 0.963 | 0.922 | 1.0 | 0.596 | 0.917 |
| two_scale-s43 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.332 |  |
| two_scale-s43 | margin | test | Agree | 782 | 0.621 | 0.953 | 0.936 | 1.0 | 0.596 |  |
| two_scale-s43 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.332 |  |
| two_scale-s43 | margin | test_heldout | Agree | 165 | 0.493 | 0.922 | 0.871 | 1.0 | 0.596 | 0.917 |
| two_scale-s43 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.332 |  |
| two_scale-s44 | majority2 | test | Agree | 726 | 0.595 | 0.983 | 0.971 | 0.999999 | 0.419 |  |
| two_scale-s44 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.221 |  |
| two_scale-s44 | majority2 | test_heldout | Agree | 147 | 0.433 | 0.958 | 0.912 | 0.999999 | 0.419 | 0.905 |
| two_scale-s44 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.221 |  |
| two_scale-s44 | margin | test | Agree | 565 | 0.444 | 0.971 | 0.954 | 1.0 | 0.419 |  |
| two_scale-s44 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.221 |  |
| two_scale-s44 | margin | test_heldout | Agree | 114 | 0.324 | 0.932 | 0.870 | 1.0 | 0.419 | 0.928 |
| two_scale-s44 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.221 |  |
| two_scale-s45 | majority2 | test | Agree | 580 | 0.506 | 0.987 | 0.974 | 0.999999 | 0.350 |  |
| two_scale-s45 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.249 |  |
| two_scale-s45 | majority2 | test_heldout | Agree | 117 | 0.411 | 0.963 | 0.912 | 0.999999 | 0.350 | 0.917 |
| two_scale-s45 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.249 |  |
| two_scale-s45 | margin | test | Agree | 438 | 0.368 | 0.975 | 0.956 | 1.0 | 0.350 |  |
| two_scale-s45 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.249 |  |
| two_scale-s45 | margin | test_heldout | Agree | 83 | 0.282 | 0.942 | 0.869 | 1.0 | 0.350 | 0.938 |
| two_scale-s45 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.249 |  |
| two_scale-s46 | majority2 | test | Agree | 684 | 0.587 | 0.985 | 0.973 | 1.0 | 0.539 |  |
| two_scale-s46 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.352 |  |
| two_scale-s46 | majority2 | test_heldout | Agree | 127 | 0.384 | 0.961 | 0.912 | 1.0 | 0.539 | 0.913 |
| two_scale-s46 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.352 |  |
| two_scale-s46 | margin | test | Agree | 684 | 0.564 | 0.968 | 0.952 | 1.0 | 0.539 |  |
| two_scale-s46 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.352 |  |
| two_scale-s46 | margin | test_heldout | Agree | 127 | 0.356 | 0.918 | 0.857 | 1.0 | 0.539 | 0.913 |
| two_scale-s46 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.352 |  |
| view_heatmap | majority2 | test | Agree | 620 | 0.530 | 0.976 | 0.961 | 1.0 | 0.477 |  |
| view_heatmap | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.200 |  |
| view_heatmap | majority2 | test_heldout | Agree | 149 | 0.406 | 0.921 | 0.866 | 1.0 | 0.477 | 0.831 |
| view_heatmap | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.200 |  |
| view_heatmap | margin | test | Agree | 620 | 0.504 | 0.949 | 0.928 | 1.0 | 0.477 |  |
| view_heatmap | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.200 |  |
| view_heatmap | margin | test_heldout | Agree | 149 | 0.381 | 0.840 | 0.773 | 1.0 | 0.477 | 0.831 |
| view_heatmap | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.200 |  |
| view_heatmap-s43 | majority2 | test | Agree | 619 | 0.521 | 0.978 | 0.964 | 1.0 | 0.460 |  |
| view_heatmap-s43 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.210 |  |
| view_heatmap-s43 | majority2 | test_heldout | Agree | 127 | 0.351 | 0.938 | 0.882 | 1.0 | 0.460 | 0.866 |
| view_heatmap-s43 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.210 |  |
| view_heatmap-s43 | margin | test | Agree | 619 | 0.491 | 0.954 | 0.934 | 1.0 | 0.460 |  |
| view_heatmap-s43 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.210 |  |
| view_heatmap-s43 | margin | test_heldout | Agree | 127 | 0.325 | 0.873 | 0.804 | 1.0 | 0.460 | 0.866 |
| view_heatmap-s43 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.210 |  |
| view_heatmap-s44 | majority2 | test | Agree | 558 | 0.475 | 0.975 | 0.958 | 1.0 | 0.428 |  |
| view_heatmap-s44 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.258 |  |
| view_heatmap-s44 | majority2 | test_heldout | Agree | 102 | 0.312 | 0.923 | 0.855 | 1.0 | 0.428 | 0.837 |
| view_heatmap-s44 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.258 |  |
| view_heatmap-s44 | margin | test | Agree | 558 | 0.452 | 0.946 | 0.924 | 1.0 | 0.428 |  |
| view_heatmap-s44 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.258 |  |
| view_heatmap-s44 | margin | test_heldout | Agree | 102 | 0.293 | 0.845 | 0.763 | 1.0 | 0.428 | 0.837 |
| view_heatmap-s44 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.258 |  |
| view_heatmap-s45 | majority2 | test | Agree | 434 | 0.382 | 0.983 | 0.966 | 0.9575 | 0.000 |  |
| view_heatmap-s45 | majority2 | test | Disagree | 70 | 0.045 | 0.949 | 0.870 | 0.9975 | 0.000 |  |
| view_heatmap-s45 | majority2 | test_heldout | Agree | 79 | 0.229 | 0.925 | 0.845 | 0.9575 | 0.000 | 0.839 |
| view_heatmap-s45 | majority2 | test_heldout | Disagree | 5 | 0.004 | 1.000 | 0.566 | 0.9975 | 0.000 |  |
| view_heatmap-s45 | margin | test | Agree | 136 | 0.113 | 0.973 | 0.931 | 0.9925 | 0.000 |  |
| view_heatmap-s45 | margin | test | Disagree | 70 | 0.063 | 0.976 | 0.909 | 0.9975 | 0.000 |  |
| view_heatmap-s45 | margin | test_heldout | Agree | 14 | 0.041 | 0.909 | 0.661 | 0.9925 | 0.000 | 0.904 |
| view_heatmap-s45 | margin | test_heldout | Disagree | 5 | 0.007 | 1.000 | 0.566 | 0.9975 | 0.000 |  |
| view_heatmap-s46 | majority2 | test | Agree | 521 | 0.455 | 0.978 | 0.961 | 1.0 | 0.407 |  |
| view_heatmap-s46 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.227 |  |
| view_heatmap-s46 | majority2 | test_heldout | Agree | 120 | 0.339 | 0.915 | 0.852 | 1.0 | 0.407 | 0.821 |
| view_heatmap-s46 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.227 |  |
| view_heatmap-s46 | margin | test | Agree | 521 | 0.432 | 0.952 | 0.930 | 1.0 | 0.407 |  |
| view_heatmap-s46 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.227 |  |
| view_heatmap-s46 | margin | test_heldout | Agree | 120 | 0.319 | 0.830 | 0.753 | 1.0 | 0.407 | 0.821 |
| view_heatmap-s46 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.227 |  |
| view_heatmap-r448 | majority2 | test | Agree | 396 | 0.351 | 0.983 | 0.965 | 1.0 | 0.320 |  |
| view_heatmap-r448 | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.241 |  |
| view_heatmap-r448 | majority2 | test_heldout | Agree | 89 | 0.298 | 0.960 | 0.896 | 1.0 | 0.320 | 0.910 |
| view_heatmap-r448 | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.241 |  |
| view_heatmap-r448 | margin | test | Agree | 396 | 0.336 | 0.963 | 0.940 | 1.0 | 0.320 |  |
| view_heatmap-r448 | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.241 |  |
| view_heatmap-r448 | margin | test_heldout | Agree | 89 | 0.274 | 0.915 | 0.839 | 1.0 | 0.320 | 0.910 |
| view_heatmap-r448 | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.241 |  |
| side_info | majority2 | test | Agree | 668 | 0.587 | 0.983 | 0.970 | 0.999999 | 0.355 |  |
| side_info | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.402 |  |
| side_info | majority2 | test_heldout | Agree | 117 | 0.435 | 0.961 | 0.909 | 0.999999 | 0.355 | 0.913 |
| side_info | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.402 |  |
| side_info | margin | test | Agree | 430 | 0.368 | 0.968 | 0.946 | 1.0 | 0.355 |  |
| side_info | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.402 |  |
| side_info | margin | test_heldout | Agree | 79 | 0.279 | 0.921 | 0.840 | 1.0 | 0.355 | 0.916 |
| side_info | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.402 |  |
| side_info_nohq | majority2 | test | Agree | 703 | 0.588 | 0.981 | 0.968 | 1.0 | 0.548 |  |
| side_info_nohq | majority2 | test | Disagree | 0 | 0.000 | — | — | never | 0.396 |  |
| side_info_nohq | majority2 | test_heldout | Agree | 122 | 0.413 | 0.968 | 0.920 | 1.0 | 0.548 | 0.929 |
| side_info_nohq | majority2 | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.396 |  |
| side_info_nohq | margin | test | Agree | 703 | 0.569 | 0.959 | 0.942 | 1.0 | 0.548 |  |
| side_info_nohq | margin | test | Disagree | 0 | 0.000 | — | — | never | 0.396 |  |
| side_info_nohq | margin | test_heldout | Agree | 122 | 0.384 | 0.933 | 0.874 | 1.0 | 0.548 | 0.929 |
| side_info_nohq | margin | test_heldout | Disagree | 0 | 0.000 | — | — | never | 0.396 |  |


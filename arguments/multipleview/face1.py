_base_ = "./default.py"
OptimizationParams = dict(
    iterations=30000,
    batch_size=2,
    lambda_dssim=0.4,
    lambda_lpips=0.01
)

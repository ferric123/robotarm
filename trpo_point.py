from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.point_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite

def run_task(*_):
    env = normalize(PointEnv())
    policy = GaussianMLPPolicy(env_spec=env.spec,)
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=500,discount=0.99,step_size=0.01
    )
    algo.train()

run_experiment_lite(run_task,n_parallel=2,snapshot_mode="last",seed=1)

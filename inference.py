from env import ResumeEnv
from baseline import simple_agent

def run():
    env = ResumeEnv()
    scores = []
    total_steps = 0

    for _ in range(3):
        obs = env.reset()
        task_name = env.current_task["id"]

        print(f"[START] task={task_name}", flush=True)

        action = simple_agent(obs["resume"])
        result = env.step(action)

        score = result["info"]["score"]
        reward = result["reward"]
        steps = 1

        print(f"[STEP] step={steps} reward={reward} action={action}", flush=True)
        print(f"[END] task={task_name} score={score:.2f} steps={steps}", flush=True)

        scores.append(score)
        total_steps += steps

    avg_score = sum(scores) / len(scores)
    print(f"[START] task=aggregate", flush=True)
    print(f"[END] task=aggregate score={avg_score:.2f} steps={total_steps}", flush=True)
    print(f"Final Score: {avg_score:.2f}", flush=True)

if __name__ == "__main__":
    run()

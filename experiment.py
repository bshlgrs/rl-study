from run_dqn_atari import *


def main():
    # [BeamRider, Breakout, Enduro, Pong, Qbert, Seaquest, SpaceInvaders]
    env = get_env(3)
    session = get_session()
    agent = DQNAgent(env, session, batch_size=512)

    agent.learn(1000001)
    env.close()

if __name__ == '__main__':
    main()
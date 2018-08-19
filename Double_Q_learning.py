from pkgs import *
from Estimator import Estimator
from Processor import StateProcessor
from copy_params import *
from behavior_policy import *

def deep_q_learning(sess, env, q_estimator, target_estimator, state_processor, num_episondes, experiment_dir,
					replay_memory_size = 500000, replay_memory_init_size = 50000, update_target_estimator_every = 10000,
					discount_factor = 0.99, epsilon_start = 1.0, epsilon_end = 0.1, epsilon_decay_steps = 500000, batch_size = 32,
					record_video_every = 50):

	
	# To store transitions
	Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

	# The replay memory
	replay_memory = []

	# Make model copier object
	estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

	# Keep track of useful statistics
	stats = plotting.EpisodeStats(
								  episode_lengths = np.zeros(num_episondes),
								  episode_rewards = np.zeros(num_episondes))

	# Create directories for checkpoints and summaries
	checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
	checkpoint_path = os.path.join(checkpoint_dir, "model")
	monitor_path = os.path.join(experiment_dir, "monitor")

	if(not os.path.exists(checkpoint_dir)):
		os.makedirs(checkpoint_dir)

	if(not os.path.exists(monitor_path)):
		os.makedirs(monitor_path)


	saver = tf.train.Saver()

	# Load a previous checkpoint if we find one.
	latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

	if(latest_checkpoint):
		print("Loading model checkpoint {}...\n".format(latest_checkpoint))
		saver.restore(sess, latest_checkpoint)

	# Get the current time step.
	total_t = sess.run(tf.train.get_global_step())

	# The epsilon decay schedule
	epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

	# The policy we're following
	policy = make_epsilon_greedy_policy(q_estimator, len(VALID_ACTIONS))

	# Populate the replay memory with initial experience
	print("Populating replay memory...")
	state = env.reset()

	# A 84 x 84 frame
	state = state_processor.process(sess, state)

	# Stack first frame 4 times (84 x 84 x 4)
	state = np.stack([state] * 4, axis = 2)


	for i in range(replay_memory_init_size):

		# Start ε at 1 and anneal it to 0.1
		action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps - 1)])
		action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

		next_state, reward, done, _ = env.step(VALID_ACTIONS[action])

		# A 84 x 84 frame.
		next_state = state_processor.process(sess, next_state)
		
		# Remove 1st frame of state (84 x 84 x 4) to append next_state's frame (84 x 84 x 1)
		# to state to become new next_state (84 x 84 x 4)
		next_state = np.append(state[:, :, 1: ], np.expand_dims(next_state, 2), axis = 2)
		replay_memory.append(Transition(state, action, reward, next_state, done))


		if(done):
			state = env.reset()
			state = state_processor.process(sess, state)
			state = np.stack([state] * 4, axis = 2)

		else:
			state = next_state

	# Record videos
	env = Monitor(env, directory = monitor_path, resume = True, video_callable = lambda count: count % record_video_every == 0)


	for i_episode in range(num_episondes):

		# Save the current checkpoint
		saver.save(tf.get_default_session(), checkpoint_path)

		# Reset the environment
		state = env.reset()
		state = state_processor.process(sess, state)
		state = np.stack([state] * 4, axis = 2)
		loss = None

		# An episode in the environment
		for t in itertools.count():

			# Epsilon for this time timestep
			epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]

			# Add epsilon to tensorboard
			epsiode_summary = tf.Summary()
			epsiode_summary.value.add(simple_value = epsilon, tag = "epsilon")
			q_estimator.summary_writer.add_summary(epsiode_summary, total_t)

			# Update the target estimator
			if(total_t % update_target_estimator_every == 0):
				estimator_copy.make(sess)
				print("\nCopied model parameters to target network.")

			# Useful for debugging
			print("\rStep {} ({}) @ Episode {} / {}, loss: {}".format(t, total_t, i_episode + 1, num_episondes, loss), end = "")
			sys.stdout.flush()

			# Take a step in the environment
			action_probs = policy(sess, state, epsilon)
			action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

			next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
			next_state = state_processor.process(sess, next_state)
			next_state = np.append(state[:, :, 1: ], np.expand_dims(next_state, 2), axis = 2)

			# Output screen of episode.
			env.render()

			# If our replay memory is full, pop the first element.
			if(len(replay_memory) == replay_memory_size):
				replay_memory.pop(0)

			# Save transition to replay memory.
			replay_memory.append(Transition(state, action, reward, next_state, done))

			# Update statistics.
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t 

			# Sample minipatch from the replay memory.
			samples = random.sample(replay_memory, batch_size)
			
			states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

			# This is where double Q-learning comes in.
			q_values_next = q_estimator.predict(sess, next_states_batch)
            		best_actions = np.argmax(q_values_next, axis = 1)
            		q_values_next_target = target_estimator.predict(sess, next_states_batch)
			targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

			# Perform gradient descent update.
			states_batch = np.array(states_batch)
			loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

			if(done):
				break

			state = next_state
			total_t += 1


		# Add summaries to Tensorboard.
		epsiode_summary = tf.Summary()
		epsiode_summary.value.add(simple_value = stats.episode_rewards[i_episode], node_name = "episode_reward", tag = "episode_reward")
		epsiode_summary.value.add(simple_value = stats.episode_lengths[i_episode], node_name = "episode_length", tag = "episode_length")
		q_estimator.summary_writer.add_summary(epsiode_summary, total_t)
		q_estimator.summary_writer.flush()

		yield total_t, plotting.EpisodeStats(
											episode_lengths = stats.episode_lengths[: i_episode + 1],
											episode_rewards = stats.episode_rewards[: i_episode + 1])

	return stats



tf.reset_default_graph()

experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))
global_step = tf.Variable(0, name = 'global_step', trainable = False)

q_estimator = Estimator(scope = "q", summaries_dir = experiment_dir)
target_estimator = Estimator(scope = "target_q")

state_processor = StateProcessor()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for t, stats in deep_q_learning(sess,
									env,
									q_estimator = q_estimator,
									target_estimator = target_estimator,
									state_processor = state_processor,
									experiment_dir = experiment_dir,
									num_episondes = 1000,
									replay_memory_size = 500000,
									replay_memory_init_size = 50000,
									update_target_estimator_every = 10000,
									epsilon_start = 1.0,
									epsilon_end = 0.1,
									epsilon_decay_steps = 500000,
									discount_factor = 0.99,
									batch_size = 32):

		print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))



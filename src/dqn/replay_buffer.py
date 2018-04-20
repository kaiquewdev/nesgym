import numpy as np


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch = np.concatenate(
            [self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = np.concatenate(
            [self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask = np.array(
            [1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: np.random.randint(
            0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def sizeExists(self):
        return self.size > int()

    def getSize(self):
        return (self.sizeExists() and self.size)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        # assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.getSize())

    def plus_one_against_id(self, idx):
        return idx + 1

    def litimus_against_id(self, idx):
        '''
        Choose the side kick
        '''
        return idx - self.frame_history_len

    def rid(self, idx=0):
        return max(0,idx - 1)

    def has_observation(self, k):
        has_observations = self.has_observations()
        is_dict = type(self.obs) == dict
        is_ndarray = self.is_nparray(self.obs)
        has_key_on_list = lambda is_ndarray_cached: is_ndarray_cached and self.obs.tolist().index(k) > -1
        has_checked_observations_with_key_on_list = lambda has_observations_cached,pre_cached_ndarray: has_observations_cached and has_key_on_list(pre_cached_ndarray) 
        has_key_on_dict = lambda is_ndarray_cached, is_dict_cached: not is_ndarray_cached and (is_dict_cached and (k in self.obs))
        has_checked_observations_with_key_on_dict = lambda has_observations_cached,pre_cached_ndarray,pre_cached_dict: has_observations_cached and has_key_on_dict(pre_cached_ndarray, pre_cached_dict)
        return (has_checked_observations_with_key_on_list(has_observations,is_ndarray) or has_checked_observations_with_key_on_dict(has_observations,is_ndarray,is_dict))

    def get_observation(self,key,common=str('')):
        has_observation = self.has_observation
        has_observations = self.has_observations
        retrieve_content = lambda key_cached: has_observation(key_cached) and self.obs[key_cached]
        checked_retrievement = lambda key_cached: has_observations() and (retrieve_content(key_cached))
        return (checked_retrievement(key) or common)

    def _has_not_observations(self):
        return self.obs is None

    def has_not_observations(self):
        has_not_suffaced_observations = self._has_not_observations
        return has_not_suffaced_observations()

    def _has_observations(self):
        return self.obs is not None

    def has_observations(self):
        has_suffaced_observations = self._has_observations
        return has_suffaced_observations()

    def _has_observations_shape(self):
        checked_observations_shape = lambda: self._has_observations() and self.obs.shape
        return checked_observations_shape()

    def has_observations_shape(self):
        has_suffaced_observations_shape = self._has_observations_shape
        return has_suffaced_observations_shape()

    def is_nparray(self, lst=[]):
        return type(lst) == np.ndarray

    def observations_shape_length(self):
        typed_len = lambda: ((self.is_nparray(self.obs)) and len(self.obs.shape))
        checked_typed_len = lambda: (self.has_observations_shape() and typed_len())  
        return (checked_typed_len() or int())

    def is_observations_length_eq(self, v=2):
        observations_shape_length = self.observations_shape_length
        is_observations_shape_length_eq_value = observations_shape_length() == v
        return is_observations_shape_length_eq_value

    def _has_done_prop(self):
        return self.done is not None

    def _mod_idx(self, v):
        return v % self.size

    # def _get_done_prop_indice(self, v):
    #     return self.done[v]

    def _get_done(self, k):
        return self.done[k]

    def get_rid_observation(self, index):
        return self.get_observation(self.rid(index))

    def _encode_observation(self, idx):
        end_idx = self.plus_one_against_id(idx)  # make noninclusive
        start_idx = self.litimus_against_id(end_idx)
        is_observations_eq_two = self.is_observations_length_eq()
        has_done_prop = self._has_done_prop()
        is_lt_zero_start_idx = start_idx < 0
        is_not_eq_size = self.num_in_buffer != self.size
        get_observation = lambda end_id_cached: self.get_observation(self.rid(end_id_cached))
        is_lt_zero_start_idx_and_is_not_eq_size = is_lt_zero_start_idx and is_not_eq_size
        modulated = lambda idx_cached: self._mod_idx(idx_cached)
        has_done_prop_switcher = lambda: self._has_done_prop()
        _get_done = self._get_done
        check_and_access_done = lambda has_done_prop: has_done_prop and _get_done(mod_idx) 
        missing_context_based_on_registering = lambda: self.frame_history_len - (end_idx - start_idx)
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if is_observations_eq_two:
            return get_observation(end_id)
        # if there weren't enough frames ever in the buffer for context
        if is_lt_zero_start_idx_and_is_not_eq_size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            mod_idx = modulated(idx)
            has_done_prop = has_done_prop_switcher()
            has_done_prop_checking_presence = check_and_access_done(has_done_prop)
            # done_value = self._get_done_prop_indice(mod_idx)
            if has_done_prop_checking_presence:
                start_idx = idx + 1
        missing_context = missing_context_based_on_registering()
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        registering_context_decoupling = start_idx < 0 or missing_context > 0
        if registering_context_decoupling:
            get_observation = self.get_observation
            zeros_like_observation = lambda: np.zeros_like(get_observation(0))
            missing_contexts = lambda: range(missing_context)
            frames = [zeros_like_observation() for _ in missing_contexts()]
            idxs = lambda: range(start_idx, end_idx)
            for idx in idxs():
                size = self.size
                modulated_size = idx % size
                frames.append(get_observation(modulated_size))
            concat_frames = np.concatenate(frames, 2)
            return concat_frames
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = (int(), int())
            has_observation_shape = self._has_observations_shape
            if has_observation_shape():
                img_h, img_w = (self.obs.shape[1], self.obs.shape[2])
                limited_observations = self.obs[start_idx:end_idx]
                transposed_observations = limited_observations.transpose(1, 2, 0, 3)
                reshaped_transposed_observations = transposed_observations.reshape(img_h, img_w, -1)
                return reshaped_transposed_observations
            else:
                return int()

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs = np.empty(
                [self.size] + list(frame.shape), dtype=np.uint8)
            self.action = np.empty([self.size],
                                   dtype=np.int32)
            self.reward = np.empty([self.size],
                                   dtype=np.float32)
            self.done = np.empty([self.size],
                                 dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done

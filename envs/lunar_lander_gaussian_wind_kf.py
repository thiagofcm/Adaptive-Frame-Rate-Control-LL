__credits__ = ["Andrea PIERRÉ"]

import math
from typing import TYPE_CHECKING

import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from gymnasium.envs.registration import register


try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e


if TYPE_CHECKING:
    import pygame


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = (
    4  # The Y location of the main engine on the body of the Lander.
)

VIEWPORT_W = 600
VIEWPORT_H = 400


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class LunarLander_GaussianWind_KF(gym.Env, EzPickle):
    r"""
    Same as `LunarLander_GaussianWind` (i.i.d. Gaussian wind/turbulence process noise,
    plus i.i.d. Gaussian measurement noise on the 6 continuous observation dims), but
    with a Kalman filter (identical math to `envs/lunar_lander_var_fps_kf.py`) fusing
    a noisy reading with a predicted estimate EVERY tick, instead of returning the raw
    noisy reading directly.

    This env has no adaptive-FPS/sensing-budget mechanic at all -- there's no concept
    of a "stale" tick here, every tick is a fresh sample (the KF equivalent of
    `obs_interval=1` always). The point is purely to train the low-level nav controller
    against the same *kind* of observation (a Kalman-fused estimate, not a raw noisy
    reading) it will be fed once embedded in the adaptive-FPS+KF env at zero staleness,
    closing a train/eval distribution mismatch that exists even in that best case.

    Reward and termination are computed from the true, noiseless state exactly as in
    `LunarLander_GaussianWind` -- only the returned observation is replaced by the KF's
    fused estimate. The 2 leg-contact booleans are never run through the KF -- they're
    always the true current reading (there's no staleness here to hold them across).

    Discrete actions only (`continuous=False`) -- `_kf_thrust_delta` below only models
    the discrete-action impulse formulas, matching how this env is actually used.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = True,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        vertical_wind_power: float = 0.0,
        sensor_noise_std: float = 0.0,
    ):
        assert not continuous, (
            "LunarLander_GaussianWind_KF is discrete-actions-only -- _kf_thrust_delta "
            "only models the discrete main/side-engine impulse formulas."
        )
        EzPickle.__init__(
            self,
            render_mode,
            continuous,
            gravity,
            enable_wind,
            wind_power,
            turbulence_power,
            vertical_wind_power,
            sensor_noise_std,
        )

        assert (
            -12.0 < gravity and gravity < 0.0
        ), f"gravity (current value: {gravity}) must be between -12 and 0"
        self.gravity = gravity

        if 0.0 > wind_power or wind_power > 20.0:
            gym.logger.warn(
                f"wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})"
            )
        self.wind_power = wind_power

        if 0.0 > turbulence_power or turbulence_power > 2.0:
            gym.logger.warn(
                f"turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})"
            )
        self.turbulence_power = turbulence_power

        # std-dev of a separate i.i.d. Gaussian force applied straight up/down each tick
        # (independent draw from wind_power's horizontal one) -- perturbs (y, vy) the
        # same way wind_power perturbs (x, vx): a direct force, integrated by Box2D, so
        # it affects vy immediately and y through the resulting velocity change.
        self.vertical_wind_power = vertical_wind_power

        # std-dev of the KF's measurement noise (R = sensor_noise_std**2 * I(6)) -- the
        # noisy reading the KF fuses each tick, NOT applied unfiltered to the returned
        # observation the way it is in the plain LunarLander_GaussianWind env.
        self.sensor_noise_std = sensor_noise_std

        self.enable_wind = enable_wind

        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.lander: Box2D.b2Body | None = None
        self.particles = []

        self.prev_reward = None

        self.continuous = continuous

        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -2.5,  # x coordinate
                -2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                -10.0,
                -10.0,
                -2 * math.pi,
                -10.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                2.5,  # x coordinate
                2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                10.0,
                10.0,
                2 * math.pi,
                10.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(low, high)

        # Nop, fire left engine, main engine, right engine (discrete-only, see assert above)
        self.action_space = spaces.Discrete(4)

        self.render_mode = render_mode

        # Kalman Filter Setup. The KF's OWN internal state order is
        # [x, vx, y, vy, angle, angular_velocity] -- chosen so Q/F are simple contiguous
        # 2x2 blocks -- which is NOT the raw obs order [x, y, vx, vy, angle,
        # angular_velocity] used by `state` below. kf_perm converts between the two
        # orderings; it's a self-inverse transposition (swaps positions 1<->2), so
        # `arr[self.kf_perm]` converts either direction. (kf_x, kf_P, kf_Q) get
        # (re)initialized per-episode in reset() -- kf_Q needs lander mass/inertia, only
        # known once a body exists. Identical derivation/values to
        # envs/lunar_lander_var_fps_kf.py -- see that file's comments for the full
        # derivation of why F's off-diagonal coefficients are 1/norm**2 and 1/20, not
        # dt=1/FPS (this env's asymmetric position/velocity state scaling).
        self.kf_perm = np.array([0, 2, 1, 3, 4, 5])

        self._kf_norm_x = VIEWPORT_W / SCALE / 2
        self._kf_norm_y = VIEWPORT_H / SCALE / 2
        self.kf_F = np.eye(6)
        self.kf_F[0, 1] = 1.0 / (self._kf_norm_x ** 2)   # x from vx
        self.kf_F[2, 3] = 1.0 / (self._kf_norm_y ** 2)   # y from vy
        self.kf_F[4, 5] = 1.0 / 20.0                       # angle from angular_velocity

        self.kf_R = (sensor_noise_std ** 2) * np.eye(6)

        # Gravity's fixed per-tick contribution to (state-scaled) vy -- a known,
        # deterministic drift, not process noise. Box2D integrates gravity into
        # linearVelocity.y over dt = 1/FPS each world.Step(); state[3] scales raw
        # velocity by (VIEWPORT_H/SCALE/2)/FPS, so the delta gets the same scaling.
        self.kf_gravity_dvy = (gravity / FPS) * (VIEWPORT_H / SCALE / 2) / FPS

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        self._destroy()

        # Bug's workaround for: https://github.com/Farama-Foundation/Gymnasium/issues/728
        # Not sure why the self._destroy() is not enough to clean(reset) the total world environment elements, need more investigation on the root cause,
        # we must create a totally new world for self.reset(), or the bug#728 will happen
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create Terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        # Create Lander body
        initial_y = VIEWPORT_H / SCALE
        initial_x = VIEWPORT_W / SCALE / 2
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # Box2D-computed exact mass/inertia for this fresh body -- used by
        # _kf_thrust_delta() to convert impulses into velocity/angular-velocity deltas,
        # and below to convert wind_power/vertical_wind_power/turbulence_power (force/
        # torque std-devs) into acceleration std-devs for Q. Read once per episode
        # since a new body is created every reset (mass/inertia are actually identical
        # every episode -- same fixed density/shape -- recomputing is just cheap/simple).
        self.kf_mass = self.lander.mass
        self.kf_inertia = self.lander.inertia

        # Q, continued from __init__: each 2x2 block is the standard discretized
        # constant-velocity-model noise (q_raw * [[dt**3/3, dt**2/2],[dt**2/2, dt]]),
        # with q_raw = (force_std / mass)**2 (or torque_std / inertia for the angular
        # block) -- converting the injected force/torque's std-dev into the
        # acceleration-noise std-dev the raw-unit CV-noise formula expects -- then
        # transformed into this env's scaled state units via D @ block @ D.T, where D
        # is the same (state_pos, state_vel) scaling used by kf_F (see its comment).
        # Identical derivation to envs/lunar_lander_var_fps_kf.py.
        kf_dt = 1.0 / FPS
        cv_base = np.array([[kf_dt**3 / 3, kf_dt**2 / 2], [kf_dt**2 / 2, kf_dt]])

        def _scaled_Q_block(q_raw, s_p, s_v):
            D = np.array([[s_p, 0.0], [0.0, s_v]])
            return D @ (q_raw * cv_base) @ D.T

        q_raw_x = (self.wind_power / self.kf_mass) ** 2
        q_raw_y = (self.vertical_wind_power / self.kf_mass) ** 2
        q_raw_angle = (self.turbulence_power / self.kf_inertia) ** 2

        self.kf_Q = np.zeros((6, 6))
        self.kf_Q[0:2, 0:2] = _scaled_Q_block(q_raw_x, 1.0 / self._kf_norm_x, self._kf_norm_x / FPS)
        self.kf_Q[2:4, 2:4] = _scaled_Q_block(q_raw_y, 1.0 / self._kf_norm_y, self._kf_norm_y / FPS)
        self.kf_Q[4:6, 4:6] = _scaled_Q_block(q_raw_angle, 1.0, 20.0 / FPS)

        # Apply the initial random impulse to the lander
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        # Create Lander Legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = (
                    +0.9 - 0.5
                )  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        if self.render_mode == "human":
            self.render()

        # Step the world with 0 action to get the initial (true, noiseless) reading.
        obs, _, _, _, _ = self._physics_step(0)

        # Seed the KF from this first noiseless reading, treated as exactly known --
        # kf_P=0 means the very first returned obs is identical whether or not it goes
        # through the fusion math, so we can just return it directly below.
        self.kf_x = np.array(obs[:6], dtype=np.float64)[self.kf_perm]
        self.kf_P = np.zeros((6, 6))
        # Kalman gain / normalized innovation squared from the most recent update --
        # exposed purely for introspection/debugging, matching envs/lunar_lander_var_fps_kf.py.
        self.kf_K = None
        self.kf_nis = None
        self.kf_last_z = None

        return obs, {}

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particle):
        while self.particles and (all_particle or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def _physics_step(self, action):
        assert self.lander is not None, "You forgot to call reset()"

        # Update wind and apply to the lander
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            # Draw order (wind_mag, torque_mag, vertical_wind_mag) deliberately matches
            # envs/lunar_lander_gaussian_wind.py's and envs/lunar_lander_var_fps_kf.py's
            # exactly -- np_random.normal() just pulls the next raw number off the RNG
            # stream and scales it by whatever std you pass, so for the SAME seed, a
            # different call order assigns the same raw numbers to different physical
            # quantities in each env, silently diverging trajectories from tick 1.
            wind_mag = self.np_random.normal(0.0, self.wind_power)
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            torque_mag = self.np_random.normal(0.0, self.turbulence_power)
            self.lander.ApplyTorque(
                torque_mag,
                True,
            )

            vertical_wind_mag = self.np_random.normal(0.0, self.vertical_wind_power)
            self.lander.ApplyForceToCenter(
                (0.0, vertical_wind_mag),
                True,
            )

        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid "

        # Apply Engine Impulses

        # Tip is the (X and Y) components of the rotation of the lander.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

        # Side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # Generate two random numbers between -1/SCALE and 1/SCALE.
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if action == 2:
            # Main engine
            m_power = 1.0

            # 4 is move a bit downwards, +-2 for randomness
            # The components of the impulse to be applied by the main engine.
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )

            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(
                    3.5,  # 3.5 is here to make particle speed adequate
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse(
                    (
                        ox * MAIN_ENGINE_POWER * m_power,
                        oy * MAIN_ENGINE_POWER * m_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if action in (1, 3):
            # Orientation/Side engines
            # action = 1 is left, action = 3 is right
            direction = action - 2
            s_power = 1.0

            # The components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14
            # This causes the position of the thrust on the body of the lander to change, depending on the orientation of the lander.
            # This in turn results in an orientation dependent torque being applied to the lander.
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * SIDE_ENGINE_POWER * s_power,
                        oy * SIDE_ENGINE_POWER * s_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= (
            m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03

        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
        if not self.lander.awake:
            terminated = True
            reward = +100

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def _kf_thrust_delta(self, action, angle_estimate):
        """Deterministic (dispersion-free) velocity/angular-velocity delta this action's
        engine impulse would produce, computed from the KF's OWN angle estimate rather
        than ground truth -- part of the predict step's control term, so thrust is
        treated as known rather than folded into process noise. Mirrors
        _physics_step()'s main/side-engine impulse formulas exactly, minus the small
        random `dispersion` jitter (that residual is left unmodeled). Discrete actions
        only, matching this env's fixed action_space = Discrete(4). Identical to
        envs/lunar_lander_var_fps_kf.py's method of the same name.

        Returns (dvx, dvy, dw) in the same state-scaled units as state[2], state[3],
        state[5].
        """
        action = int(action)
        tip = (math.sin(angle_estimate), math.cos(angle_estimate))
        side = (-tip[1], tip[0])

        lever = (0.0, 0.0)
        impulse = (0.0, 0.0)

        if action == 2:
            # Main engine
            ox0 = tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE)
            oy0 = -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE)
            lever = (ox0, oy0)
            impulse = (-ox0 * MAIN_ENGINE_POWER, -oy0 * MAIN_ENGINE_POWER)
        elif action in (1, 3):
            # Side engines: action=1 -> left (direction=-1), action=3 -> right (direction=+1)
            direction = action - 2
            ox0 = side[0] * (direction * SIDE_ENGINE_AWAY / SCALE)
            oy0 = -side[1] * (direction * SIDE_ENGINE_AWAY / SCALE)
            lever = (ox0 - tip[0] * 17 / SCALE, oy0 + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)
            impulse = (-ox0 * SIDE_ENGINE_POWER, -oy0 * SIDE_ENGINE_POWER)
        # action == 0: no thrust, lever/impulse stay (0,0)

        dv_raw_x = impulse[0] / self.kf_mass
        dv_raw_y = impulse[1] / self.kf_mass
        # 2D cross product r x F -- angular impulse from the off-center lever arm.
        torque_impulse = lever[0] * impulse[1] - lever[1] * impulse[0]
        dw_raw = torque_impulse / self.kf_inertia

        dvx = dv_raw_x * (VIEWPORT_W / SCALE / 2) / FPS
        dvy = dv_raw_y * (VIEWPORT_H / SCALE / 2) / FPS
        dw = dw_raw * 20.0 / FPS
        return dvx, dvy, dw

    def step(self, action):
        assert self.lander is not None

        obs, reward, terminated, truncated, info = self._physics_step(action)

        # --- Kalman filter predict (every tick) ---
        # Thrust direction uses the KF's OWN current angle estimate (index 4 -- the
        # same position in both orderings, unaffected by kf_perm), not ground truth,
        # computed from the action that was just physically applied. There's no frozen
        # sub-controller in this env -- the trainee IS the controller, so `action` here
        # is exactly the engine action that drove _physics_step above.
        dvx, dvy, dw = self._kf_thrust_delta(action, self.kf_x[4])
        kf_b = np.array([0.0, dvx, 0.0, dvy + self.kf_gravity_dvy, 0.0, dw])
        self.kf_x = self.kf_F @ self.kf_x + kf_b
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q

        # --- Kalman filter update (unconditional -- every tick is fresh here) ---
        # No obs_interval/staleness concept in this env at all: unlike
        # envs/lunar_lander_var_fps_kf.py, there is no stale-tick branch -- the update
        # below always runs. Guarded like the sibling envs' own sensor_noise_std check
        # -- np_random.normal(scale=0.0, size=6) still consumes 6 numbers from the RNG
        # stream even though they get multiplied by 0, so skipping the draw entirely at
        # sensor_noise_std=0 keeps "same seed" comparable across the sibling envs.
        noise = self.np_random.normal(0.0, self.sensor_noise_std, size=6) if self.sensor_noise_std > 0.0 else np.zeros(6)
        z = (obs[:6] + noise)[self.kf_perm]
        self.kf_last_z = z.copy()
        innovation = z - self.kf_x
        S = self.kf_P + self.kf_R
        S_inv = np.linalg.inv(S)
        K = self.kf_P @ S_inv
        self.kf_x = self.kf_x + K @ innovation
        self.kf_P = (np.eye(6) - K) @ self.kf_P
        self.kf_K = K
        # Normalized innovation squared -- a consistency check independent of NEES
        # (uses the pre-update predicted P via S, not the post-update fused P).
        # Expected value under a correctly-calibrated filter: state dim (6).
        self.kf_nis = float(innovation.T @ S_inv @ innovation)

        # KF-fused estimate (6 continuous dims, converted back to raw obs order) + true
        # current leg-contact booleans (never run through the KF).
        fused_obs = np.concatenate([
            self.kf_x[self.kf_perm].astype(np.float32),
            obs[6:8],
        ])

        return fused_obs, reward, terminated, False, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )

        self._clean_particles(False)

        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )

                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True
                    )

                for x in [self.helipad_x1, self.helipad_x2]:
                    x = x * SCALE
                    flagy1 = self.helipad_y * SCALE
                    flagy2 = flagy1 + 50
                    pygame.draw.line(
                        self.surf,
                        color=(255, 255, 255),
                        start_pos=(x, flagy1),
                        end_pos=(x, flagy2),
                        width=1,
                    )
                    pygame.draw.polygon(
                        self.surf,
                        color=(204, 204, 0),
                        points=[
                            (x, flagy2),
                            (x, flagy2 - 10),
                            (x + 25, flagy2 - 5),
                        ],
                    )
                    gfxdraw.aapolygon(
                        self.surf,
                        [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                        (204, 204, 0),
                    )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


register(
    id="LunarLander_GaussianWind_KF",
    entry_point="envs.lunar_lander_gaussian_wind_kf:LunarLander_GaussianWind_KF",
)

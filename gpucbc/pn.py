import numpy as np

PI = np.pi


def _zero_function(*args, **kwargs):
    return 0


def taylor_f2_amplitude_0(eta, chi_1, chi_2, mass_ratio):
    return 1


taylor_f2_amplitude_1 = _zero_function


def taylor_f2_amplitude_2(eta, chi_1, chi_2, mass_ratio):
    return -323 / 224 + 451 * eta / 168


def taylor_f2_amplitude_3(eta, chi_1, chi_2, mass_ratio):
    return chi_1 * (27 * mass_ratio / 16 - 11 * eta / 12 + 27 / 16) + chi_2 * (
        -27 * mass_ratio / 16 - 11 * eta / 12 + 27 / 16
    )


def taylor_f2_amplitude_4(eta, chi_1, chi_2, mass_ratio):
    return (
        chi_1 ** 2 * (-81 * mass_ratio / 64 + 81 * eta / 32 - 81 / 64)
        + chi_2 ** 2 * (81 * mass_ratio / 64 + 81 * eta / 32 - 81 / 64)
        + (105271 / 24192 * eta ** 2 - 1975055 / 338688 * eta - 27312085 / 8128512)
        - 47 / 16 * eta * chi_1 * chi_2
    )


def taylor_f2_amplitude_5(eta, chi_1, chi_2, mass_ratio):
    return (
        chi_1 ** 3 * (mass_ratio * (3 / 16 - 3 * eta / 16) - 9 * eta / 16 + 3 / 16)
        + chi_1
        * (
            mass_ratio * (287213 / 32256 - 2083 * eta / 8064)
            - 2227 * eta ** 2 / 2016
            - 15569 * eta / 1344
            + 287213 / 32256
        )
        + chi_2 ** 3 * (mass_ratio * (3 * eta / 16 - 3 / 16) - 9 * eta / 16 + 3 / 16)
        + chi_2
        * (
            mass_ratio * (2083 * eta / 8064 - 287213 / 32256)
            - 2227 * eta ** 2 / 2016
            - 15569 * eta / 1344
            + 287213 / 32256
        )
        - 85 * PI / 64
        + 85 * PI * eta / 16
    )


def taylor_f2_amplitude_6(eta, chi_1, chi_2, mass_ratio):
    return (
        chi_1 * (-17 * PI * mass_ratio / 12 + 5 * PI * eta / 3 - 17 * PI / 12)
        + chi_2 * (17 * PI * mass_ratio / 12 + 5 * PI * eta / 3 - 17 * PI / 12)
        + chi_1 * chi_2 * (-133249 * eta ** 2 / 8064 - 319321 * eta / 32256)
        + chi_1 ** 2
        * (
            mass_ratio * (-14139 * eta / 32256 - 49039 / 14336)
            + 163199 * eta ** 2 / 16128
            + 158633 * eta / 64512
            - 49039 / 14336
        )
        + chi_2 ** 2
        * (
            mass_ratio * (14139 * eta / 32256 + 49039 / 14336)
            + 163199 * eta ** 2 / 16128
            + 158633 * eta / 64512
            - 49039 / 14336
        )
        - 177520268561 / 8583708672
        + (545384828789 / 5007163392 - 205 * PI ** 2 / 48) * eta
        - 3248849057 * eta ** 2 / 178827264
        + 34473079 * eta ** 3 / 6386688
    )


def taylor_f2_phase_0(args):
    return 1


taylor_f2_phase_1 = _zero_function


def taylor_f2_phase_2(args):
    return 55 * args.eta / 9 + 3715 / 756


def taylor_f2_phase_3(args):
    phase = -16 * PI
    for m_on_m, chi in zip([args.m1_on_m, args.m2_on_m], [args.chi_1, args.chi_2]):
        phase += m_on_m * (25 + 38 / 3 * m_on_m) * chi
    return phase


def taylor_f2_phase_4(args):
    phase = 15293365 / 508032 + 27145 / 504 * args.eta + 3085 / 72 * args.eta ** 2
    phase -= 395 / 4 * args.eta * args.chi_1 * args.chi_2
    for m_on_m, chi, qm_def in zip(
        [args.m1_on_m, args.m2_on_m],
        [args.chi_1, args.chi_2],
        [args.qm_def_1, args.qm_def_2],
    ):
        phase -= (50 * qm_def + 5 / 8) * m_on_m ** 2 * chi ** 2
    return phase


def taylor_f2_phase_5(args):
    phase = 5 / 9 * (7729 / 84 - 13 * args.eta) * PI
    for m_on_m, chi in zip([args.m1_on_m, args.m2_on_m], [args.chi_1, args.chi_2]):
        phase -= (
            chi
            * m_on_m
            * (
                13915 / 84
                - m_on_m * (1 - m_on_m) * 10 / 3
                + m_on_m * (12760 / 81 + m_on_m * (1 - m_on_m) * 170 / 9)
            )
        )
    return phase


def taylor_f2_phase_6(args):
    phase = (
        11583231236531 / 4694215680
        - 640 / 3 * PI ** 2
        - 6848 / 21 * np.euler_gamma
        + args.eta * (-15737765635 / 3048192 + 2255 / 12 * PI ** 2)
        + args.eta ** 2 * 76055 / 1728
        - args.eta ** 3 * 127825 / 1296
        + taylor_f2_phase_6l(args) * np.log(4)
    )
    phase += (32675 / 112 + 5575 / 18 * args.eta) * args.eta * args.chi_1 * args.chi_2
    for m_on_m, chi, qm_def in zip(
        [args.m1_on_m, args.m2_on_m],
        [args.chi_1, args.chi_2],
        [args.qm_def_1, args.qm_def_2],
    ):
        phase += PI * m_on_m * (1490 / 3 + m_on_m * 260) * chi
        phase += (
            (47035 / 84 + 2935 / 6 * m_on_m - 120 * m_on_m ** 2)
            * m_on_m ** 2
            * qm_def
            * chi ** 2
        )
        phase += (
            (-410825 / 672 - 1085 / 12 * m_on_m + 1255 / 36 * m_on_m ** 2)
            * m_on_m ** 2
            * chi ** 2
        )
    return phase


def taylor_f2_phase_7(args):
    phase = PI * (
        77096675 / 254016 + 378515 / 1512 * args.eta - 74045 / 756 * args.eta ** 2
    )
    for m_on_m, chi in zip([args.m1_on_m, args.m2_on_m], [args.chi_1, args.chi_2]):
        phase += (
            chi
            * m_on_m
            * (
                -170978035 / 48384
                + args.eta * 2876425 / 672
                + args.eta ** 2 * 4735 / 144
                + m_on_m
                * (
                    -7189233785 / 1524096
                    + args.eta * 458555 / 3024
                    - args.eta ** 2 * 5345 / 72
                )
            )
        )
    return phase


taylor_f2_phase_8 = _zero_function
taylor_f2_phase_9 = _zero_function


def taylor_f2_phase_10(args):
    phase = 0
    for lambda_, m_on_m in zip(
        [args.lambda_1, args.lambda_2], [args.m1_on_m, args.m2_on_m]
    ):
        phase += (-288 + 264 * m_on_m) * m_on_m ** 4 * lambda_
    return phase


taylor_f2_phase_11 = _zero_function


def taylor_f2_phase_12(args):
    phase = 0
    for lambda_, m_on_m in zip(
        [args.lambda_1, args.lambda_2], [args.m1_on_m, args.m2_on_m]
    ):
        phase += (
            (
                -15895 / 28
                + 4595 / 28 * m_on_m
                + 5715 / 14 * m_on_m ** 2
                - 325 / 7 * m_on_m ** 3
            )
            * m_on_m ** 4
            * lambda_
        )
    return phase


def taylor_f2_phase_13(args):
    phase = 0
    for lambda_, m_on_m in zip(
        [args.lambda_1, args.lambda_2], [args.m1_on_m, args.m2_on_m]
    ):
        phase += 24 * (12 - 11 * m_on_m) * PI * m_on_m ** 4 * lambda_
    return phase


def taylor_f2_phase_14(args):
    phase = 0
    for lambda_, m_on_m in zip(
        [args.lambda_1, args.lambda_2], [args.m1_on_m, args.m2_on_m]
    ):
        phase += (
            -(m_on_m ** 4)
            * lambda_
            * 24
            * (
                39927845 / 508032
                - 480043345 / 9144576 * m_on_m
                + 9860575 / 127008 * m_on_m ** 2
                - 421821905 / 2286144 * m_on_m ** 3
                + 4359700 / 35721 * m_on_m ** 4
                - 10578445 / 285768 * m_on_m ** 5
            )
        )
    return phase


def taylor_f2_phase_15(args):
    phase = 0
    for lambda_, m_on_m in zip(
        [args.lambda_1, args.lambda_2], [args.m1_on_m, args.m2_on_m]
    ):
        phase += (
            m_on_m ** 4
            * 1
            / 28
            * PI
            * lambda_
            * (27719 - 22127 * m_on_m + 7022 * m_on_m ** 2 - 10232 * m_on_m ** 3)
        )
    return phase


taylor_f2_phase_0l = _zero_function
taylor_f2_phase_1l = _zero_function
taylor_f2_phase_2l = _zero_function
taylor_f2_phase_3l = _zero_function
taylor_f2_phase_4l = _zero_function


def taylor_f2_phase_5l(args):
    return taylor_f2_phase_5(args) * 3


def taylor_f2_phase_6l(args):
    return -6848 / 21


taylor_f2_phase_7l = _zero_function
taylor_f2_phase_8l = _zero_function

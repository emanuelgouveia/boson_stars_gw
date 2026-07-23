# Modified from Tiago Fernandes, 2024
# Extended for Boson Star (BBS) signal injection
# BBS waveforms from Evstafyeva et al. (2024), arXiv:2406.02715
# github.com/tamaraevst/Boson-star-waveforms

import argparse
import glob
import json
import os
import warnings
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from pycbc.detector import Detector
from pycbc.types import TimeSeries as PyCBCTimeSeries
from pycbc.filter import matched_filter
from pycbc.distributions.angular import UniformSolidAngle
from scipy.signal import resample
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from gwpy.timeseries import TimeSeries
import h5py as h5
import romspline
from kuibit import gw_utils

warnings.filterwarnings("ignore")


# PHYSICAL CONSTANTS
# These are needed to convert the NR waveform from geometric units (G=c=1) into physical units (seconds, metres)
M_SUN_SECONDS = 4.925491e-6   # 1 solar mass in seconds
MPC_IN_SECONDS = 1.029e14     # 1 Megaparsec in light-seconds


# NumpyEncoder
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types
    https://github.com/hmallen/numpyencoder
    """
    def default(self, obj):
        int_types = (
            np.int_, np.intc, np.intp,
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
        )
        if isinstance(obj, int_types):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)



def _get_Hlm(l, m, filename, sr=1):

    # Open file and read the time axes of the splines
    f       = h5.File(filename, "r")
    t_amp   = f[f"amp_l{l}_m{m}"]["X"][:]
    t_phase = f[f"phase_l{l}_m{m}"]["X"][:]
    end     = max(t_amp[-1], t_phase[-1])   # end time in geometric units
    f.close()

    # Read the spline objects and evaluate on a uniform grid
    spline_amp   = romspline.readSpline(filename, f"amp_l{l}_m{m}")
    spline_phase = romspline.readSpline(filename, f"phase_l{l}_m{m}")

    t     = np.linspace(0, end, int(sr * end))
    amp   = spline_amp(t)
    phase = spline_phase(t)

    # Combine amplitude and phase into complex waveform mode
    H_lm = amp * np.exp(1j * phase)

    return H_lm, t



def gen_data_strain_bs(mass, inclination, distance, filename, sample_rate=4096):

    
    phi_used = float(np.random.uniform(0, 2 * np.pi))

    
    l = 2
    H = None   # accumulator for the summed complex waveform
    t = None   # geometric time array

    for m in range(-l, l + 1):
        H_lm, t_lm = _get_Hlm(l, m, filename)

        if H is None:
            # First mode — set up the accumulator
            H = np.zeros(len(H_lm), dtype=complex)
            t = t_lm

        # Trim to common length in case modes differ by 1 sample
        min_len = min(len(H), len(H_lm))
        H     = H[:min_len]
        t     = t[:min_len]
        H_lm  = H_lm[:min_len]

    
        H += H_lm * gw_utils.sYlm(-2, l, m, inclination, phi_used)

    
    mass_in_seconds     = mass * M_SUN_SECONDS
    distance_in_seconds = distance * MPC_IN_SECONDS


    t_physical  = t * mass_in_seconds                           #Verificar contas
    amp_scale   = mass_in_seconds / distance_in_seconds         #Verificar contas
    
    hp_physical = H.real * amp_scale
    hc_physical = -H.imag * amp_scale

    #Resample
    n_target     = int(t_physical[-1] * sample_rate)
    hp_resampled = resample(hp_physical, n_target) 
    hc_resampled = resample(hc_physical, n_target) 

    # Wrap in PyCBC TimeSeries with the correct time step
    dt = 1.0 / sample_rate
    hp = PyCBCTimeSeries(hp_resampled.astype(np.float64), delta_t=dt)
    hc = PyCBCTimeSeries(hc_resampled.astype(np.float64), delta_t=dt)

    return hp, hc, phi_used


class Generator:

    def __init__(
        self,
        work_dir: str = "outputs_bbs_10k",
        noise_h1: str = "/projects/F202509140CPCAA1/boson_stars_gw/classificador-bs/ruido/O3a_Noise_H1.hdf5",
        noise_l1: str = "/projects/F202509140CPCAA1/boson_stars_gw/classificador-bs/ruido/O3a_Noise_L1.hdf5",
        noise_v1: str = "/projects/F202509140CPCAA1/boson_stars_gw/classificador-bs/ruido/O3a_Noise_V1.hdf5",
        bs_waveform_dir: str = None,
        thread: int = 0
    ):
        # Set up output directories
        self.work_dir   = work_dir
        self.config_dir = os.path.join(self.work_dir, "config")
        self.data_dir   = os.path.join(self.work_dir, "sig")
        self.bg_dir     = os.path.join(self.work_dir, "bg")
        self.model_dir  = os.path.join(self.work_dir, "model")
        self.thread     = thread

        for folder in (self.work_dir, self.config_dir, self.data_dir,
                       self.bg_dir, self.model_dir):
            os.makedirs(folder, exist_ok=True)

        # Check noise files exist then load them
        for noise_file in (noise_h1, noise_l1, noise_v1):
            assert os.path.isfile(noise_file), f"File {noise_file} does not exist"

        self.noise_h1, self.noise_l1, self.noise_v1 = self.load_noise_TS(
            noise_h1, noise_l1, noise_v1
        )
        self.sample_rate = self.noise_h1.sample_rate.value
        self.solid_angle_sampler = UniformSolidAngle()

        # Build list of available NR waveform files
        # If no directory given, BBS mode is not available
        if bs_waveform_dir is not None:
            self.bs_waveform_library = glob.glob(
                os.path.join(bs_waveform_dir, "*.h5")
            )
            assert len(self.bs_waveform_library) > 0, \
                f"No .h5 files found in {bs_waveform_dir}"
            print(f"Found {len(self.bs_waveform_library)} BS waveform files.")
        else:
            self.bs_waveform_library = None

   
    @property
    def configuration(self) -> dict:
        rng = self.rng

        # Total mass — freely variable via geometric unit rescaling
        total_mass = rng.uniform(50, 500)

        # Distance and sky location — same as BBH
        distance     = rng.integers(10, 10000)
        detector_ref = rng.choice(["H1", "L1", "V1"])

        sky_location    = self.solid_angle_sampler.rvs(1)[0]
        declination     = np.pi/2 - sky_location["theta"]
        right_ascension = sky_location["phi"]

        # Inclination — polar viewing angle, feeds into sYlm in gen_data_strain_bs
        # phi (azimuthal angle) is sampled inside gen_data_strain_bs and logged
        angles      = self.solid_angle_sampler.rvs(1)[0]
        inclination = angles["theta"]

        # Pick a random NR waveform file from the library
        waveform_path = str(rng.choice(self.bs_waveform_library))

        return {
            "sample_rate":     self.sample_rate,
            "delta_t":         1.0 / self.sample_rate,
            "total_mass":      total_mass,
            "distance":        distance,
            "inclination":     inclination,
            "declination":     declination,
            "polarization":    None,          # set inside gen_data_strain_bs
            "right_ascension": right_ascension,
            "detector_ref":    detector_ref,
            "waveform_path":   waveform_path,
            "signal_type":     "BBS",
        }


    @staticmethod
    def load_noise_TS(path1, path2, path3):
        assert (
            os.path.isfile(path1) &
            os.path.isfile(path2) &
            os.path.isfile(path3)
        ), "One of the noise files does not exist"
        return (
            TimeSeries.read(path1),
            TimeSeries.read(path2),
            TimeSeries.read(path3),
        )

    @staticmethod
    def _save_sample(path_to_save, filename, data_1, ts_1, ts_2, ts_3, times, **kwargs):
        savepath = os.path.join(path_to_save, filename)
        np.savez_compressed(
            savepath, qgraph=data_1,
            ts_h1=ts_1, ts_l1=ts_2, ts_v1=ts_3, times=times,
            **kwargs # This ensures w_h1, w_l1, etc. are saved
        )

    @staticmethod
    def _write_json(data: dict, file: str, mode="w") -> None:
        with open(file, mode) as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)

    @staticmethod
    def get_asd_hlv(ts_h, ts_l, ts_v, length=2048) -> Tuple[list]:
        fft_length1 = int(max(2, np.ceil(length * ts_h.dt.decompose().value)))
        fft_length2 = int(max(2, np.ceil(length * ts_l.dt.decompose().value)))
        fft_length3 = int(max(2, np.ceil(length * ts_v.dt.decompose().value)))

        asd_h1 = ts_h.asd(overlap=0, fftlength=fft_length1, window="hann", method="welch")
        asd_h1 = asd_h1.interpolate(1.0 / ts_h.duration.decompose().value)
        asd_l1 = ts_l.asd(overlap=0, fftlength=fft_length2, window="hann", method="welch")
        asd_l1 = asd_l1.interpolate(1.0 / ts_l.duration.decompose().value)
        asd_v1 = ts_v.asd(overlap=0, fftlength=fft_length3, window="hann", method="welch")
        asd_v1 = asd_v1.interpolate(1.0 / ts_v.duration.decompose().value)
        return asd_h1, asd_l1, asd_v1

    @staticmethod
    def get_psd(ts, length=2048) -> list:
        fft_length = int(max(2, np.ceil(length * ts.dt.decompose().value)))
        psd = ts.psd(overlap=0, fftlength=fft_length, window="hann", method="welch")
        psd = psd.interpolate(1.0 / ts.duration.decompose().value)
        return psd

    @staticmethod
    def whiten_hlv(ts_h, ts_l, ts_v, asd_h, asd_l, asd_v) -> Tuple[list]:
        ts_h = ts_h.whiten(asd=asd_h).bandpass(20, 300).notch(60).notch(120).notch(240)
        ts_l = ts_l.whiten(asd=asd_l).bandpass(20, 300).notch(60).notch(120).notch(240)
        ts_v = ts_v.whiten(asd=asd_v).bandpass(20, 300).notch(50).notch(100).notch(200)
        return ts_h, ts_l, ts_v

    @staticmethod
    def get_qtransform(ts_h, ts_l, ts_v, time_window, t_res, f_res) -> Tuple[list]:
        q_transforms = (
            ts.q_transform(
                outseg=time_window,
                tres=t_res,
                norm="median",
                frange=(20, 300),   
                fres=f_res,
            ).value
            for ts in (ts_h, ts_l, ts_v)
        )
        return q_transforms

    
    def _create_samples(self, idx):
        self.rng = np.random.default_rng()
        det_names = ("H1", "L1", "V1")

        dets = {det_name: Detector(det_name) for det_name in det_names}

        SNR_MIN  = 8.0
        accepted = False
        attempt  = 0

        while not accepted:
            attempt += 1
            bgs = {"H1": self.noise_h1, "L1": self.noise_l1, "V1": self.noise_v1}

            config = self.configuration
            try:
                hp, hc, phi_used = gen_data_strain_bs(
                    mass        = config["total_mass"],
                    inclination = config["inclination"],
                    distance    = config["distance"],
                    filename    = config["waveform_path"],
                    sample_rate = self.sample_rate,
                )
                config["polarization"] = phi_used
            except RuntimeError:
                continue
            except Exception as exception:
                print(f"\n\nException: {type(exception).__name__}\n\n")
                continue

            offset = self.rng.integers(
                5 * self.sample_rate,
                len(bgs["H1"]) - len(hp) - 5 * self.sample_rate,
                endpoint=True
            )
            t0 = bgs["H1"].times[offset].value
            hp.start_time = hc.start_time = t0
            config["t0"] = t0

            tdelays = {}
            for d in dets.values():
                dt = d.time_delay_from_detector(
                    Detector(config["detector_ref"]),
                    config["right_ascension"],
                    config["declination"],
                    t0,
                )
                tdelays[d.name] = dt

            models          = dict()
            models_whitened = dict()

            for det_name in det_names:
                models[det_name] = dets[det_name].project_wave(
                    hp, hc,
                    config["right_ascension"],
                    config["declination"],
                    config["polarization"],
                )
                models[det_name] = TimeSeries.from_pycbc(models[det_name])

            t_max = models["H1"].times.value[models["H1"].argmax()]
            t_max = t_max - self.rng.random() * 2 * 0.1 + 0.1

            sigs          = dict()
            sigs_whitened = dict()

            for det_name in det_names:
                bgs[det_name]  = bgs[det_name].crop(t_max - 4, t_max + 4)
                sigs[det_name] = bgs[det_name].inject(models[det_name])
                sigs[det_name].shift(f"{tdelays[det_name]}s")

            asd_h, asd_l, asd_v = self.get_asd_hlv(
                bgs["H1"], bgs["L1"], bgs["V1"], length=self.sample_rate
            )
            sigs_whitened["H1"], sigs_whitened["L1"], sigs_whitened["V1"] = self.whiten_hlv(
                sigs["H1"], sigs["L1"], sigs["V1"], asd_h, asd_l, asd_v
            )

            for det_name in det_names:
                sig_length   = len(sigs[det_name])
                model_length = len(models[det_name])
                if model_length > sig_length:
                    models[det_name] = models[det_name][model_length - sig_length:]
                elif model_length < sig_length:
                    pad_before = (sig_length - model_length) // 2
                    pad_after  = sig_length - model_length - pad_before
                    models[det_name] = models[det_name].pad((pad_before, pad_after))

            models_whitened["H1"], models_whitened["L1"], models_whitened["V1"] = self.whiten_hlv(
                models["H1"], models["L1"], models["V1"], asd_h, asd_l, asd_v
            )

            snrs_matchfilter = {}
            snrs_correlate   = {}
            mf_highpass_f    = 20

            for det_name in det_names:
                signal = sigs[det_name].highpass(mf_highpass_f)
                model  = models[det_name].highpass(mf_highpass_f)
                psd    = self.get_psd(
                    bgs[det_name].highpass(mf_highpass_f), length=self.sample_rate
                )
                model_fft  = model.fft().to_pycbc()
                signal_fft = signal.fft().to_pycbc()

                assert len(model_fft) == len(signal_fft) == len(psd), \
                    f"Shape mismatch: model_fft={len(model_fft)}, " \
                    f"signal_fft={len(signal_fft)}, psd={len(psd)}"

                snr = matched_filter(
                    model_fft, signal_fft, psd.to_pycbc(),
                    low_frequency_cutoff=mf_highpass_f
                )
                snrs_matchfilter[det_name] = abs(snr).max()
                snrs_correlate[det_name]   = (
                    sigs_whitened[det_name].correlate(models_whitened[det_name]).max().value
                )

            config["snr_matchfilter"] = np.sqrt(
                np.sum([snrs_matchfilter[d] ** 2 for d in det_names])
            )
            config["snr_correlate"] = np.sqrt(
                np.sum([snrs_correlate[d] ** 2 for d in det_names])
            )

            if config["snr_matchfilter"] < SNR_MIN:
                print(f"  [sample {idx}, attempt {attempt}] SNR={config['snr_matchfilter']:.2f} < {SNR_MIN}, resampling...")
                continue

            x = y = 275
            time_window = (t_max - 0.28, t_max + 0.28)
            tres = abs(-t_max + 0.28 + t_max + 0.28) / x
            fres = 280 / y

            qg_H1, qg_L1, qg_V1 = self.get_qtransform(
                *sigs.values(), time_window, tres, fres,
            )
            qg_data = np.stack([qg_H1, qg_L1, qg_V1])

            bg_qg_H1, bg_qg_L1, bg_qg_V1 = self.get_qtransform(
                *bgs.values(), time_window, tres, fres,
            )
            bg_qg_data = np.stack([bg_qg_H1, bg_qg_L1, bg_qg_V1])

            self._write_json(config, self.config_dir + f"/{idx}_config.json")
            self._save_sample(
                self.data_dir, f"{idx}_sample", qg_data,
                sigs["H1"].value,
                sigs["L1"].value,
                sigs["V1"].value,
                sigs["H1"].times.value - t_max,
                w_h1=sigs_whitened["H1"].value,
                w_l1=sigs_whitened["L1"].value,
                w_v1=sigs_whitened["V1"].value
            )
            self._save_sample(
                self.bg_dir, f"{idx}_bg", bg_qg_data,
                bgs["H1"].value, bgs["L1"].value, bgs["V1"].value,
                bgs["H1"].times.value
            )
            self._save_sample(
                self.model_dir, f"{idx}_model", None,
                models_whitened["H1"].value, models_whitened["L1"].value, models_whitened["V1"].value,
                models_whitened["H1"].times.value
            )

            accepted = True

   
    def run_gen(self, n: int, n_jobs: int = 4) -> None:
        print("Generating BBS samples...")
        print(f"Number of jobs: {n_jobs}")

        # TODO: TEMPORARY fix — parallel disabled due to shared noise TimeSeries
        n_jobs = 1
        if n_jobs > 1:
            Parallel(n_jobs=n_jobs)(
                delayed(self._create_samples)(idx=i) for i in tqdm(range(1, n + 1))
            )
        else:
            for i in tqdm(range(1, n + 1)):
                self._create_samples(idx=i)


def _normalize_qgraph(qgraph):
    qt_h1, qt_l1, qt_v1 = qgraph
    qt_h1 = np.uint8((qt_h1 - qt_h1.min()) * 255 / qt_h1.max())
    qt_l1 = np.uint8((qt_l1 - qt_l1.min()) * 255 / qt_l1.max())
    qt_v1 = np.uint8((qt_v1 - qt_v1.min()) * 255 / qt_v1.max())
    img = np.stack([qt_h1, qt_l1, qt_v1])
    img = Image.fromarray(img.T)
    return img


def generate_img(obj_path: str):
    obj    = np.load(obj_path)
    qgraph = _normalize_qgraph(obj["qgraph"])
    plt.imshow(qgraph, origin="lower", aspect="auto")
    plt.colorbar()
    plt.show()


def plot_sample(sample_path: str, output_dir: str = "plots"):
    data  = np.load(sample_path)
    times = data['times']

    plt.figure(figsize=(15, 10))
    plt.subplot(3, 2, (1, 5))
    plt.imshow(_normalize_qgraph(data['qgraph']), origin="lower", aspect="auto")
    plt.title('Q-transform Image')
    plt.colorbar()

    for i, (key, title) in enumerate(
        [('ts_h1', 'H1'), ('ts_l1', 'L1'), ('ts_v1', 'V1')]
    ):
        plt.subplot(3, 2, 2 + i * 2)
        plt.plot(times, data[key])
        plt.title(f'{title} Time Series')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir,
        os.path.basename(sample_path).replace('.npz', '.png')))
    plt.close()


def compute_snr(signal, noise_psd):
    signal_fft = np.fft.fft(signal)
    snr = np.sqrt(4 * np.sum(np.abs(signal_fft)**2 / noise_psd) / len(signal))
    return float(snr)



def create_parser():
    parser = argparse.ArgumentParser(description="Generate BBS GW injection data.")
    parser.add_argument("-n",    "--num_files",      metavar="N",    type=int, nargs="?", default=10000)
    parser.add_argument("-w",    "--work-dir",       metavar="PATH", type=str, nargs="?", default="outputs_bbs_random_10k")
    parser.add_argument("-t",    "--thread",         metavar="T",    type=int, nargs="?", default=1)
    parser.add_argument("-nh1",  "--noise_hanford",  metavar="NH1",  type=str, nargs="?",
        default="/projects/F202509140CPCAA1/boson_stars_gw/classificador-bs/ruido/O3a_Noise_H1.hdf5")
    parser.add_argument("-nl1",  "--noise_livingston", metavar="NL1", type=str, nargs="?",
        default="/projects/F202509140CPCAA1/boson_stars_gw/classificador-bs/ruido/O3a_Noise_L1.hdf5")
    parser.add_argument("-nv1",  "--noise-virgo",    metavar="NV1",  type=str, nargs="?",
        default="/projects/F202509140CPCAA1/boson_stars_gw/classificador-bs/ruido/O3a_Noise_V1.hdf5")
    parser.add_argument("-bsdir","--bs_waveform_dir",metavar="BSDIR",type=str, nargs="?",
    default="/projects/F202509140CPCAA1/boson_stars_gw/classificador-bs/waveforms/GRChombo",
    help="Directory containing NR boson star .h5 waveform files.")
    parser.add_argument("-s",    "--seed",           metavar="S",    type=int, nargs="?", default=0)
    parser.add_argument("-v",    "--visualize",      action="store_true")
    parser.add_argument("-nc", "--num_cores", metavar="NC", type=int, nargs="?", default=1)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args   = parser.parse_args()

    print(f"Arguments: {args}")

    if args.visualize:
        generate_img(f"{args.work_dir}/sig/1_sample.npz")
    else:
        gen = Generator(
            work_dir        = args.work_dir,
            noise_h1        = args.noise_hanford,
            noise_l1        = args.noise_livingston,
            noise_v1        = args.noise_virgo,
            bs_waveform_dir = args.bs_waveform_dir,
            thread          = args.thread,
        )
        gen.run_gen(n=args.num_files, n_jobs=args.num_cores)

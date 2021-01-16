"""tests for vak.files.spect module"""
from pathlib import Path

import vak.files
from vak.constants import VALID_AUDIO_FORMATS


def test_find_audio_fname_with_mat(spect_list_mat):
    """test ```vak.files.spect.find_audio_fname`` works when we give it a list of """
    audio_fnames = [vak.files.spect.find_audio_fname(spect_path)
                    for spect_path in spect_list_mat]
    for mat_spect_path, audio_fname in zip(spect_list_mat, audio_fnames):
        # make sure we gout out a filename that was actually in spect_path
        assert audio_fname in mat_spect_path
        # make sure it's some valid audio format
        assert Path(audio_fname).suffix.replace('.', '') in VALID_AUDIO_FORMATS


def test_find_audio_fname_with_npz(spect_list_npz):
    audio_fnames = [vak.files.spect.find_audio_fname(spect_path)
                    for spect_path in spect_list_npz]
    for npz_spect_path, audio_fname in zip(spect_list_npz, audio_fnames):
        # make sure we gout out a filename that was actually in spect_path
        assert audio_fname in npz_spect_path
        assert Path(audio_fname).suffix.replace('.', '') in VALID_AUDIO_FORMATS

"""fixtures relating to array files containing spectrograms"""
import pytest

import vak.files.spect


@pytest.fixture
def spect_dir_mat(source_test_data_root):
    return source_test_data_root.joinpath('spect_mat_annot_yarden', 'llb3', 'spect')


@pytest.fixture
def spect_list_mat(spect_dir_mat):
    return sorted(spect_dir_mat.glob('*.mat'))


@pytest.fixture
def spect_dir_npz(generated_test_data_root):
    return sorted(generated_test_data_root.joinpath('prep',
                                                    'train',
                                                    'audio_cbin_annot_notmat').glob('spectrograms_generated*'))[0]


@pytest.fixture
def spect_list_npz(spect_dir_npz):
    return sorted(spect_dir_npz.glob('*.spect.npz'))


@pytest.fixture
def spect_list_mat_all_labels_in_labelset(spect_list_mat,
                                          annot_list_yarden,
                                          labelset_yarden):
    """list of .mat spectrogram files where all labels in associated annotation **are** in labelset"""
    labelset_yarden = set(labelset_yarden)
    spect_list_labels_in_labelset = []
    for spect_path in spect_list_mat:
        audio_fname = vak.files.spect.find_audio_fname(spect_path)
        annot = [annot for annot in annot_list_yarden if annot.audio_path.name == audio_fname]
        assert len(annot) == 1
        annot = annot[0]
        if set(annot.seq.labels).issubset(labelset_yarden):
            spect_list_labels_in_labelset.append(spect_path)

    return spect_list_labels_in_labelset


@pytest.fixture
def spect_list_mat_labels_not_in_labelset(spect_list_mat,
                                          annot_list_yarden,
                                          labelset_yarden):
    """list of .mat spectrogram files where some labels in associated annotation are **not** in labelset"""
    labelset_yarden = set(labelset_yarden)
    spect_list_labels_not_in_labelset = []
    for spect_path in spect_list_mat:
        audio_fname = vak.files.spect.find_audio_fname(spect_path)
        annot = [annot for annot in annot_list_yarden if annot.audio_path.name == audio_fname]
        assert len(annot) == 1
        annot = annot[0]
        # notice if labels **not** a subset of labelset
        if not set(annot.seq.labels).issubset(labelset_yarden):
            spect_list_labels_not_in_labelset.append(spect_path)

    err = 'not finding .mat spectrogram files where labels in associated annotations are not in dataset'
    assert len(spect_list_labels_not_in_labelset) > 0, err
    return spect_list_labels_not_in_labelset


@pytest.fixture
def spect_list_npz_all_labels_in_labelset(spect_list_npz,
                                          annot_list_notmat,
                                          labelset_notmat):
    """list of .npz spectrogram files where all labels in associated annotation **are** in labelset"""
    labelset_notmat = set(labelset_notmat)
    spect_list_labels_in_labelset = []
    for spect_path in spect_list_npz:
        audio_fname = vak.files.spect.find_audio_fname(spect_path)
        annot = [annot for annot in annot_list_notmat if annot.audio_path.name == audio_fname]
        assert len(annot) == 1
        annot = annot[0]
        if set(annot.seq.labels).issubset(labelset_notmat):
            spect_list_labels_in_labelset.append(spect_path)

    return spect_list_labels_in_labelset


@pytest.fixture
def spect_list_npz_labels_not_in_labelset(spect_list_npz,
                                          annot_list_notmat,
                                          labelset_notmat):
    """list of .npz spectrogram files where some labels in associated annotation are  **not** in labelset"""
    labelset_notmat = set(labelset_notmat)
    spect_list_labels_not_in_labelset = []
    for spect_path in spect_list_npz:
        audio_fname = vak.files.spect.find_audio_fname(spect_path)
        annot = [annot for annot in annot_list_notmat if annot.audio_path.name == audio_fname]
        assert len(annot) == 1
        annot = annot[0]
        if set(annot.seq.labels).issubset(labelset_notmat):
            spect_list_labels_not_in_labelset.append(spect_path)

    err = 'not finding .npz spectrogram files where labels in associated annotations are not in dataset'
    assert len(spect_list_labels_not_in_labelset) > 0, err
    return spect_list_labels_not_in_labelset

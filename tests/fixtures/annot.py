"""fixtures relating to annotation files"""
import crowsetta
import pytest
import toml


@pytest.fixture
def annot_file_yarden(source_test_data_root):
    return source_test_data_root.joinpath('spect_mat_annot_yarden', 'llb3', 'llb3_annot_subset.mat')


@pytest.fixture
def annot_list_yarden(annot_file_yarden):
    scribe = crowsetta.Transcriber(format='yarden')
    annot_list = scribe.from_file(annot_file_yarden)
    return annot_list


@pytest.fixture
def labelset_yarden():
    """labelset as it would be loaded from a toml file

    don't return a set because we need to use this to test functions that convert it to a set
    """
    return [str(an_int) for an_int in [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]]


@pytest.fixture
def annot_list_notmat(source_test_data_root):
    annot_notmat = sorted(
        source_test_data_root.joinpath('audio_cbin_annot_notmat',
                                       'gy6or6',
                                       '032312').glob('*.not.mat')
    )
    scribe = crowsetta.Transcriber(format='notmat')
    annot_list = scribe.from_file(annot_notmat)
    return annot_list


@pytest.fixture
def labelset_notmat(generated_test_configs_root):
    """labelset as it would be loaded from a toml file

    don't return a set because we need to use this to test functions that convert it to a set"""
    a_train_notmat_config = sorted(
        generated_test_configs_root.glob('*train*notmat*toml')
    )[0]  # get first config.toml from glob list
    # doesn't really matter which config, they all have labelset
    with a_train_notmat_config.open('r') as fp:
        a_train_notmat_toml = toml.load(fp)
    labelset = a_train_notmat_toml['PREP']['labelset']
    return labelset

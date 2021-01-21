"""tests for vak.io.dataframe module"""
import pandas as pd

import vak.io.dataframe

from .test_spect import expected_spect_paths_in_dataframe


def test_from_files_with_audio_cbin(audio_dir_cbin,
                                    default_spect_params,
                                    labelset_notmat,
                                    spect_list_npz_all_labels_in_labelset,
                                    spect_list_npz_labels_not_in_labelset,
                                    ):
    """test that ``vak.io.dataframe.from_files`` works
    when we point it at directory of .cbin audio files
    and specify an annotation format"""
    vak_df = vak.io.dataframe.from_files(data_dir=audio_dir_cbin,
                                         labelset=labelset_notmat,
                                         annot_format='notmat',
                                         audio_format='cbin',
                                         spect_format=None,
                                         annot_file=None,
                                         spect_params=default_spect_params)

    assert isinstance(vak_df, pd.DataFrame)
    assert expected_spect_paths_in_dataframe(vak_df,
                                             expected_spect_paths=spect_list_npz_all_labels_in_labelset,
                                             not_expected_spect_paths=spect_list_npz_labels_not_in_labelset)


def test_from_files_with_audio_cbin_no_annot(audio_dir_cbin,
                                             default_spect_params,
                                             labelset_notmat):
    """test that ``vak.io.dataframe.from_files`` works
    when we point it at directory of .cbin audio files
    and  **do not** specify an annotation format"""
    vak_df = vak.io.dataframe.from_files(data_dir=audio_dir_cbin,
                                         annot_format=None,
                                         labelset=None,
                                         audio_format='cbin',
                                         spect_format=None,
                                         annot_file=None,
                                         spect_params=default_spect_params)

    assert isinstance(vak_df, pd.DataFrame)


def test_from_files_with_audio_cbin_no_labelset(audio_dir_cbin,
                                                default_spect_params):
    """test that ``vak.io.dataframe.from_files`` works
    when we point it at directory of .cbin audio files
    and specify an annotation format"""
    vak_df = vak.io.dataframe.from_files(data_dir=audio_dir_cbin,
                                         annot_format='notmat',
                                         labelset=None,
                                         audio_format='cbin',
                                         spect_format=None,
                                         annot_file=None,
                                         spect_params=default_spect_params)

    assert isinstance(vak_df, pd.DataFrame)


def test_from_files_with_spect_mat(spect_dir_mat,
                                   labelset_yarden,
                                   annot_file_yarden):
    """test that ``vak.io.dataframe.from_files`` works
    when we point it at directory of .mat array files
    and specify an annotation format"""
    vak_df = vak.io.dataframe.from_files(data_dir=spect_dir_mat,
                                         labelset=labelset_yarden,
                                         annot_format='yarden',
                                         audio_format=None,
                                         spect_format='mat',
                                         annot_file=annot_file_yarden,
                                         spect_params=None)

    assert isinstance(vak_df, pd.DataFrame)


def test_from_files_with_spect_mat_no_annot(spect_dir_mat):
    """test that ``vak.io.dataframe.from_files`` works
    when we point it at directory of .mat array files
    and **do not** specify an annotation format"""
    vak_df = vak.io.dataframe.from_files(data_dir=spect_dir_mat,
                                         labelset=None,
                                         annot_format=None,
                                         audio_format=None,
                                         spect_format='mat',
                                         annot_file=None,
                                         spect_params=None)

    assert isinstance(vak_df, pd.DataFrame)


def test_from_files_with_spect_mat_no_labelset(spect_dir_mat,
                                               labelset_yarden,
                                               annot_file_yarden,
                                               annot_list_yarden):
    """test that ``vak.io.dataframe.from_files`` works
    when we point it at directory of .mat array files
    and specify an annotation format
    but do not specify a labelset"""
    vak_df = vak.io.dataframe.from_files(data_dir=spect_dir_mat,
                                         labelset=None,
                                         annot_format='yarden',
                                         audio_format=None,
                                         spect_format='mat',
                                         annot_file=annot_file_yarden,
                                         spect_params=None)

    assert isinstance(vak_df, pd.DataFrame)


def test_add_split_col(audio_dir_cbin,
                       default_spect_params,
                       labelset_notmat,
                       tmp_path):
    """test that ``add_split_col`` adds a 'split' column
    to a DataFrame, where all values in the Series are the
    specified split (a string)"""
    vak_df = vak.io.dataframe.from_files(data_dir=audio_dir_cbin,
                                         labelset=labelset_notmat,
                                         annot_format='notmat',
                                         audio_format='cbin',
                                         spect_format=None,
                                         annot_file=None,
                                         spect_params=default_spect_params)

    assert 'split' not in vak_df.columns

    vak_df = vak.io.dataframe.add_split_col(vak_df, split='train')
    assert 'split' in vak_df.columns

    assert vak_df['split'].unique().item() == 'train'

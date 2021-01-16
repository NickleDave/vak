"""tests for ``vak.io.spect`` module

main thing we test here is that functions in this module work
with both ``mat`` and ``npz`` array files containing spectrograms

i.e the annotation format matters less, just have to test with both ``mat`` and ``npz``

---- NOTE ABOUT TESTS HERE ----
a lot of these unit tests are cut-and-paste where only the fixtures change,
(from ``mat`` to ``npz``) violating the principle of Don't Repeat Yourself.
It would seem natural to parametrize these tests
but the problem is that, as far as I can tell, there's no robust way to
parametrize with fixtures in ``pytest``.
There's a proposal to add this:
https://docs.pytest.org/en/stable/proposals/parametrize_with_fixtures.html
and a plug-in that makes it possible, but looks not regularly maintained?
https://github.com/TvoroG/pytest-lazy-fixture

Note also for all tests that use `labelset`:
there are example songs in both the `yarden` annotation and `notmat` annotation
where there are labels **not** in the labelset.
The `expected` spect list fixtures have those songs removed.
In other words these tests do properly test whether passing in the labelset
works as expected, causing the function to filter out those songs.
"""
import copy
from pathlib import Path

import pandas as pd
import pytest

import vak.io.spect
import vak.files.spect


def expected_spect_paths_in_dataframe(vak_df,
                                      expected_spect_paths,
                                      not_expected_spect_paths=None):
    """tests that a dataframe ``vak_df`` contains
    all paths in ``expected_spect_paths``, and only those paths,
    in its ``spect_path`` column.
    If so, returns True.

    Parameters
    ----------
    vak_df : pandas.Dataframe
        created by vak.io.spect.to_dataframe
    expected_spect_paths : list
        of paths to spectrogram files, that **should** be in vak_df.spect_path column
    not_expected_spect_paths : list
        of paths to spectrogram files, that should **not** be in vak_df.spect_path column
    """
    assert type(vak_df) == pd.DataFrame

    spect_paths_from_df = [Path(spect_path) for spect_path in vak_df['spect_path']]

    for expected_spect_path in list(expected_spect_paths):
        assert expected_spect_path in spect_paths_from_df
        spect_paths_from_df.remove(expected_spect_path)

    # test that **only** expected paths were in DataFrame
    if not_expected_spect_paths is not None:
        for not_expected_spect_path in  not_expected_spect_paths:
            assert not_expected_spect_path not in spect_paths_from_df

    # test that **only** expected paths were in DataFrame
    # spect_paths_from_df should be empty after popping off all the expected paths
    assert len(spect_paths_from_df) == 0  # yes I know this isn't "Pythonic". It's readable, go away.

    return True  # all asserts passed


def test_to_dataframe_spect_dir_mat_annot_yarden(spect_dir_mat,
                                                 spect_list_mat_all_labels_in_labelset,
                                                 spect_list_mat_labels_not_in_labelset,
                                                 labelset_yarden,
                                                 annot_list_yarden):
    """test that ``vak.io.spect.to_dataframe`` works
    when we point it at directory + give it list of annotations"""
    vak_df = vak.io.spect.to_dataframe(spect_format='mat',
                                       spect_dir=spect_dir_mat,
                                       labelset=labelset_yarden,
                                       annot_list=annot_list_yarden)
    assert expected_spect_paths_in_dataframe(vak_df,
                                              spect_list_mat_all_labels_in_labelset,
                                              spect_list_mat_labels_not_in_labelset)


def test_to_dataframe_spect_dir_npz_annot_notmat(spect_dir_npz,
                                                 spect_list_npz_all_labels_in_labelset,
                                                 spect_list_npz_labels_not_in_labelset,
                                                 labelset_notmat,
                                                 annot_list_notmat):
    """test that ``vak.io.spect.to_dataframe`` works when we point it at directory + give it list of annotations"""
    vak_df = vak.io.spect.to_dataframe(spect_format='npz',
                                       spect_dir=spect_dir_npz,
                                       labelset=labelset_notmat,
                                       annot_list=annot_list_notmat)
    assert expected_spect_paths_in_dataframe(vak_df,
                                             spect_list_npz_all_labels_in_labelset,
                                             spect_list_npz_labels_not_in_labelset)


def test_to_dataframe_spect_dir_mat_annot_yarden_no_labelset(spect_dir_mat,
                                                             spect_list_mat,
                                                             annot_list_yarden):
    """test that ``vak.io.spect.to_dataframe`` works when we point it at directory + give it list of annotations
    but do not give it a labelset to filter out files"""
    vak_df = vak.io.spect.to_dataframe(spect_format='mat',
                                       spect_dir=spect_dir_mat,
                                       labelset=None,
                                       annot_list=annot_list_yarden)
    assert expected_spect_paths_in_dataframe(vak_df, spect_list_mat)


def test_to_dataframe_spect_dir_npz_annot_notmat_no_labelset(spect_dir_npz,
                                                             spect_list_npz,
                                                             annot_list_notmat):
    """test that ``vak.io.spect.to_dataframe`` works when we point it at directory + give it list of annotations
    but do not give it a labelset to filter out files"""
    vak_df = vak.io.spect.to_dataframe(spect_format='npz',
                                       spect_dir=spect_dir_npz,
                                       labelset=None,
                                       annot_list=annot_list_notmat)
    assert expected_spect_paths_in_dataframe(vak_df, spect_list_npz)


def test_to_dataframe_spect_dir_mat_without_annot(spect_dir_mat, spect_list_mat):
    """test ``vak.io.spect.to_dataframe`` works with a dataset from spectrogram files without annotations,
    # e.g. if we're going to predict the annotations using the spectrograms"""
    vak_df = vak.io.spect.to_dataframe(spect_format='mat',
                                       spect_dir=spect_dir_mat,
                                       annot_list=None)
    assert expected_spect_paths_in_dataframe(vak_df, spect_list_mat)


def test_to_dataframe_spect_dir_npz_without_annot(spect_dir_npz, spect_list_npz):
    """test ``vak.io.spect.to_dataframe`` works with a dataset from spectrogram files without annotations,
    # e.g. if we're going to predict the annotations using the spectrograms"""
    vak_df = vak.io.spect.to_dataframe(spect_format='npz',
                                       spect_dir=spect_dir_npz,
                                       annot_list=None)
    assert expected_spect_paths_in_dataframe(vak_df, spect_list_npz)


def test_to_dataframe_spect_files_mat_annot_yarden(spect_list_mat,
                                                   labelset_yarden,
                                                   annot_list_yarden,
                                                   spect_list_mat_all_labels_in_labelset,
                                                   spect_list_mat_labels_not_in_labelset,
                                                   ):
    """test that ``vak.io.spect.to_dataframe`` works
    when we give it list of spectrogram files and a list of annotations"""
    vak_df = vak.io.spect.to_dataframe(spect_format='mat',
                                       spect_files=spect_list_mat,
                                       labelset=labelset_yarden,
                                       annot_list=annot_list_yarden)
    assert expected_spect_paths_in_dataframe(vak_df,
                                              spect_list_mat_all_labels_in_labelset,
                                              spect_list_mat_labels_not_in_labelset)


def test_to_dataframe_spect_files_npz_annot_notmat(spect_list_npz,
                                                   labelset_notmat,
                                                   annot_list_notmat,
                                                   spect_list_npz_all_labels_in_labelset,
                                                   spect_list_npz_labels_not_in_labelset,
                                                   ):
    """test that ``vak.io.spect.to_dataframe`` works
    when we give it list of spectrogram files and a list of annotations"""
    vak_df = vak.io.spect.to_dataframe(spect_format='npz',
                                       spect_files=spect_list_npz,
                                       labelset=labelset_notmat,
                                       annot_list=annot_list_notmat)
    assert expected_spect_paths_in_dataframe(vak_df,
                                              spect_list_npz_all_labels_in_labelset,
                                              spect_list_npz_labels_not_in_labelset,
                                              )


def test_to_dataframe_spect_files_mat_annot_yarden_no_labelset(spect_list_mat,
                                                               annot_list_yarden):
    """test that ``vak.io.spect.to_dataframe`` works
    when we give it list of spectrogram files and a list of annotations
    but do not give it a labelset to filter out files"""
    # need to copy since it's mutable but we need the original unmutated
    # when we call expected_spect_paths_in_dataframe.
    # For some reason I haven't figured out, ``to_dataframe`` **does** change the list.
    spect_list_mat_copy = copy.deepcopy(spect_list_mat)
    vak_df = vak.io.spect.to_dataframe(spect_format='mat',
                                       spect_files=spect_list_mat_copy,
                                       labelset=None,
                                       annot_list=annot_list_yarden)
    assert expected_spect_paths_in_dataframe(vak_df, spect_list_mat)


def test_to_dataframe_spect_files_npz_annot_notmat_no_labelset(spect_list_npz,
                                                               annot_list_notmat):
    """test that ``vak.io.spect.to_dataframe`` works
    when we give it list of spectrogram files and a list of annotations
    but do not give it a labelset to filter out files"""
    # need to copy since it's mutable but we need the original unmutated
    # when we call expected_spect_paths_in_dataframe.
    # For some reason I haven't figured out, ``to_dataframe`` **does** change the list.
    spect_list_npz_copy = copy.deepcopy(spect_list_npz)
    vak_df = vak.io.spect.to_dataframe(spect_format='npz',
                                       spect_files=spect_list_npz_copy,
                                       labelset=None,
                                       annot_list=annot_list_notmat)
    assert expected_spect_paths_in_dataframe(vak_df, spect_list_npz)


def test_to_dataframe_spect_mat_annot_yarden_map(spect_list_mat,
                                                 labelset_yarden,
                                                 annot_list_yarden,
                                                 spect_list_mat_all_labels_in_labelset,
                                                 spect_list_mat_labels_not_in_labelset,
                                                 ):
    """test that ``vak.io.spect.to_dataframe`` works
    when we give it a dict that maps spectrogram files to annotations
    but do not give it a labelset to filter out files"""
    spect_annot_map = dict(zip(spect_list_mat, annot_list_yarden))
    vak_df = vak.io.spect.to_dataframe(spect_format='mat',
                                       labelset=labelset_yarden,
                                       spect_annot_map=spect_annot_map)
    assert expected_spect_paths_in_dataframe(vak_df,
                                              spect_list_mat_all_labels_in_labelset,
                                              spect_list_mat_labels_not_in_labelset,
                                              )


def test_to_dataframe_spect_npz_annot_notmat_map(spect_list_npz,
                                                 labelset_notmat,
                                                 annot_list_notmat,
                                                 spect_list_npz_all_labels_in_labelset,
                                                 spect_list_npz_labels_not_in_labelset):
    """test that ``vak.io.spect.to_dataframe`` works
    when we give it a dict that maps spectrogram files to annotations
    but do not give it a labelset to filter out files"""
    spect_annot_map = dict(zip(spect_list_npz, annot_list_notmat))
    vak_df = vak.io.spect.to_dataframe(spect_format='npz',
                                       labelset=labelset_notmat,
                                       spect_annot_map=spect_annot_map)
    assert expected_spect_paths_in_dataframe(vak_df,
                                              spect_list_npz_all_labels_in_labelset,
                                              spect_list_npz_labels_not_in_labelset,
                                              )


def test_to_dataframe_spect_mat_annot_yarden_map_no_labelset(spect_list_mat,
                                                             annot_list_yarden):
    """test that ``vak.io.spect.to_dataframe`` works
    when we give it a dict that maps spectrogram files to annotations
    but do not give it a labelset to filter out files"""
    spect_annot_map = dict(zip(spect_list_mat, annot_list_yarden))
    vak_df = vak.io.spect.to_dataframe(spect_format='mat',
                                       labelset=None,
                                       spect_annot_map=spect_annot_map)
    assert expected_spect_paths_in_dataframe(vak_df, spect_list_mat)


def test_to_dataframe_spect_npz_annot_notmat_map_no_labelset(spect_list_npz,
                                                             annot_list_notmat):
    """test that ``vak.io.spect.to_dataframe`` works
    when we give it a dict that maps spectrogram files to annotations
    but do not give it a labelset to filter out files"""
    spect_annot_map = dict(zip(spect_list_npz, annot_list_notmat))
    vak_df = vak.io.spect.to_dataframe(spect_format='npz',
                                       labelset=None,
                                       spect_annot_map=spect_annot_map)
    assert expected_spect_paths_in_dataframe(vak_df, spect_list_npz)


def test_to_dataframe_no_spect_dir_files_or_map_raises(annot_list_yarden):
    """test that calling ``to_dataframe`` without one of:
    spect dir, spect files, or spect files/annotations mapping
    raises ValueError"""
    with pytest.raises(ValueError):
        vak.io.spect.to_dataframe(spect_format='mat',
                                  spect_dir=None,
                                  spect_files=None,
                                  annot_list=annot_list_yarden,
                                  spect_annot_map=None)


def test_to_dataframe_invalid_spect_format_raises(spect_dir_mat,
                                                  annot_list_yarden):
    """test that calling ``to_dataframe`` with an invalid spect format raises a ValueError"""
    with pytest.raises(ValueError):
        vak.io.spect.to_dataframe(spect_format='npy',  # 'npy' not a valid spect format
                                  spect_dir=spect_dir_mat,
                                  annot_list=annot_list_yarden)


def test_to_dataframe_dir_and_list_raises(spect_dir_mat,
                                          spect_list_mat,
                                          annot_list_yarden):
    """test that calling ``to_dataframe`` with both dir and list raises a ValueError"""
    with pytest.raises(ValueError):
        vak.io.spect.to_dataframe(spect_format='mat',
                                  spect_dir=spect_dir_mat,
                                  spect_files=spect_list_mat,
                                  annot_list=annot_list_yarden)


def test_to_dataframe_dir_and_map_raises(spect_dir_mat,
                                          spect_list_mat,
                                          annot_list_yarden):
    """test that calling ``to_dataframe`` with both dir and map raises a ValueError"""
    spect_annot_map = dict(zip(spect_list_mat, annot_list_yarden))
    with pytest.raises(ValueError):
        vak.io.spect.to_dataframe(spect_format='mat',
                                  spect_dir=spect_dir_mat,
                                  spect_annot_map=spect_annot_map)


def test_to_dataframe_list_and_map_raises(spect_dir_mat,
                                          spect_list_mat,
                                          annot_list_yarden):
    """test that calling ``to_dataframe`` with both list and map raises a ValueError"""
    spect_annot_map = dict(zip(spect_list_mat, annot_list_yarden))
    with pytest.raises(ValueError):
        vak.io.spect.to_dataframe(spect_format='mat',
                                  spect_files=spect_list_mat,
                                  spect_annot_map=spect_annot_map)


def test_to_dataframe_annot_list_and_map_raises(spect_dir_mat,
                                                spect_list_mat,
                                                annot_list_yarden):
    """test that calling ``to_dataframe`` with both list of annotations and map raises a ValueError"""
    spect_annot_map = dict(zip(spect_list_mat, annot_list_yarden))
    with pytest.raises(ValueError):
        vak.io.spect.to_dataframe(spect_format='mat',
                                  spect_annot_map=spect_annot_map,
                                  annot_list=annot_list_yarden)

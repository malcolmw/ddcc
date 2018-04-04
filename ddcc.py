"""
This script was tested with Python2.7.14, because the core dependency
(pyasdf) is not yet (04/02/2018) stable under python3.

TODO:: logging
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import ConfigParser as configparser
import h5py
import logging
import numpy as np
import obspy as op
import obspy.signal.cross_correlation
import os
import pandas as pd
import pyasdf
import sys
import time
import traceback

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("wfs_in", type=str, help="input ASDF waveform dataset.")
    parser.add_argument("events_in", type=str, help="input event/phase data "\
                                                    "HDFStore")
    parser.add_argument("corr_out", type=str, help="output HDF5 file for "\
                                                   "correlation results")
    parser.add_argument("config_file", type=str, help="configuration file")
    parser.add_argument('-l', '--logfile', type=str, help='log file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    not_found = [f for f in (args.wfs_in,
                             args.events_in,
                             args.config_file) if not os.path.isfile(f)]
    if len(not_found) > 0:
        raise(IOError("file(s) not found: {:s}".format(", ".join(not_found))))
    return(args)

def main(args, cfg):
    logger.info("starting process")
    with pyasdf.ASDFDataSet(args.wfs_in, mode="r") as asdf_dset:

        comm, rank, size, MPI = asdf_dset.mpi

        if rank == 0:
            logger.info("loading events to scatter")
            df0_event, df0_phase = load_event_data(args.events_in)
            logger.info("events loaded")
            data = np.array_split(df0_event.index, size)
        else:
            data = None
        logger.info("receiving scattered data")
        data = comm.scatter(data, root=0)

        logger.info("loading event data to process")
        df0_event, df0_phase = load_event_data(args.events_in, evids=data)

        with h5py.File(args.corr_out, "w", driver="mpio", comm=comm) as f5out:
            logger.info("initializing output file")
            initialize_f5out(f5out, cfg)
            for evid in data:
                try:
                    correlate(evid, asdf_dset, f5out, df0_event, df0_phase, cfg)
                except Exception as err:
                    logger.error(traceback.print_tb(sys.exc_info()[0]))
                    logger.error(traceback.print_tb(sys.exc_info()[1]))
                    logger.error(traceback.print_tb(sys.exc_info()[2]))
    logger.info("process completed successfully")

def parse_config(config_file):
    parser = configparser.ConfigParser()
    parser.readfp(open(config_file))
    config = {"tlead_p"  : parser.getfloat("general", "tlead_p"),
              "tlead_s"  : parser.getfloat("general", "tlead_s"),
              "tlag_p"   : parser.getfloat("general", "tlag_p"),
              "tlag_s"   : parser.getfloat("general", "tlag_s"),
              "corr_min" : parser.getfloat("general", "corr_min"),
              "ncorr_min": parser.getint(  "general", "ncorr_min"),
              "knn"      : parser.getint(  "general", "knn")}
    return(config)


def configure_logging(verbose, logfile):
    """
    A utility function to configure logging.
    """
    if verbose is True:
        level = logging.DEBUG
    else:
        level = logging.INFO
    for name in (__name__,):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if level == logging.DEBUG:
            formatter = logging.Formatter(fmt="%(asctime)s::%(levelname)s::"\
                    "%(funcName)s()::%(process)d:: %(message)s",
                    datefmt="%Y%j %H:%M:%S")
        else:
            formatter = logging.Formatter(fmt="%(asctime)s::%(levelname)s::"\
                    " %(message)s",
                    datefmt="%Y%j %H:%M:%S")
        if logfile is not None:
            file_handler = logging.FileHandler(logfile)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

def load_event_data(f5in, evids=None):
    with pd.HDFStore(f5in) as cat:
        if evids is None:
            return(cat["event"], cat["phase"])
        else:
            return(cat["event"].loc[evids], cat["phase"].loc[evids])



def get_knn(evid, df_event, k=10):
    """
    Get the K nearest-neighbour events.
    Only returns events with IDs greater than evid.

    evid :: event ID of primary event.
    k    :: number of nearest-neighbours to retrieve.
    """
    lat0, lon0, depth0, time0, orid0 = df_event.loc[evid]
    _df = df_event.copy()
    _df = _df[_df.index >= evid]
    _df["distance"] = np.sqrt(
        (np.square(_df["lat"]-lat0) + np.square(_df["lon"]-lon0))*111.11**2
        +np.square(_df["depth"]-depth0)
    )
    return(_df.sort_values("distance").iloc[:k+1])

def get_phases(evids, df_phase):
    """
    Get phase data for a set of events.

    evids :: list of event IDs to retrieve phase data for.
    """
    return(
        df_phase[
            df_phase.index.isin(evids)
        ].sort_index(
        ).drop_duplicates(
            ["sta", "iphase"]
        ).sort_values(["sta", "iphase"])
    )

def initialize_f5out(f5out, cfg):
    """
    Initialize metadata structure for output HDF5 file.

    This is a collective operation.
    """
    with pd.HDFStore(args.events_in, "r") as f5in:
        df0_event = f5in["event"]
        df0_phase = f5in["phase"]
        for evid0 in df0_event.index:
            for evidB in get_knn(evid0, df0_event, k=cfg["knn"]).iloc[1:].index:
                for _, arrival in get_phases((evid0, evidB), df0_phase).iterrows():
                    grpid = "{:d}/{:d}/{:s}".format(evid0, evidB, arrival["sta"])
                    if grpid not in f5out:
                        grp = f5out.create_group(grpid)
                    else:
                        grp = f5out[grpid]
                    dset = grp.create_dataset(arrival["iphase"],
                                              (2,),
                                              dtype="f",
                                              fillvalue=np.nan)
                    dset.attrs["chan"] = arrival["chan"]

def correlate(evid, asdf_dset, f5out, df0_event, df0_phase, cfg):
    """
    Correlate an event with its K nearest-neighbours.

    Arguments:
    evid      :: int
                 The event ID of the "control" or "template" event.
    asdf_dset :: pyasdf.ASDFDataSet
                 The waveform dataset. It is assumed that each waveform
                 is given a tag "event$EVID" where $EVID is the event ID
                 of the associated event. This tag format may change in
                 the future.
    f5out     :: h5py.File
                 The output data file where results will be stored. This
                 file needs to be initialized with the proper metadata
                 structure; this can be achieved with initialize_f5out().
    df0_event :: pandas.DataFrame
                 A DataFrame of all events in the dataset. The DataFrame
                 must be indexed by event ID and the columns must be
                 lat, lon, depth, time, and orid - I think orid can be
                 left out though.
    df0_phase :: pandas.DataFrame
                 A DataFrame of all phase information in the dataset. The
                 DataFrame must be indexed by event ID and the columns must
                 be arid, orid, sta, chan, iphase, time, prefor, and snet -
                 I think orid and prefor can be left out.

    Returns:
    None
    """
    # df_event :: DataFrame of K-nearest-neighbour events including
    #             primary event.
    df_event = get_knn(evid, df0_event)
    df_phase = get_phases(df_event.index, df0_phase)
    # event0 :: primary event
    # evid0  :: primary event ID
    event0 = df_event.iloc[0]
    evid0 = event0.name
    for evidB, eventB in df_event.iloc[1:].iterrows():
        # log_tstart :: for logging elapsed time
        log_tstart = time.time()
        # ot0 :: origin time of the primary event
        # otB :: origin time of the secondary event
        ot0 = op.core.UTCDateTime(event0["time"])
        otB = op.core.UTCDateTime(eventB["time"])

        # _df_phase :: DataFrame with arrival data for the primary and
        #              secondary events
        _df_phase = get_phases((evid0, evidB), df_phase=df_phase)

        # measurements :: storage for double-difference and average
        #                 cross-correlation coefficient
        measurements = {}
        for _, arrival in _df_phase.iterrows():
            # dbldiff :: array of double-difference measurements for
            #            this station:phase pair
            # ccmax   :: array of maximum cross-correlation
            #            coefficients for this station:phase pair
            dbldiff, ccmax = [], []
            try:
                # st0 :: waveform Stream for primary event
                # stB :: waveform Stream for secondary event
                st0 = asdf_dset.waveforms["%s.%s" % (arrival["snet"],
                                                     arrival["sta"])]["event%d" % evid0]
                stB = asdf_dset.waveforms["%s.%s" % (arrival["snet"],
                                                     arrival["sta"])]["event%d" % evidB]
            except KeyError as exc:
                continue
            # tr0 :: waveform Trace for primary event
            for tr0 in st0:
                try:
                    # trB :: waveform Trace for secondary event
                    trB = stB.select(channel=tr0.stats.channel)[0]
                except IndexError:
                    continue
                # trX :: "template" trace; this is ideally the primary event Trace,
                #        but the secondary event Trace will be used if the only
                #        arrival for this station:phase pair comes from the secondary
                #        event
                # trY :: "test" trace; this is ideally the secondary event Trace
                # atX :: arrival-time of the template arrival
                # otY :: origin-time of the "test" event
                atX = op.core.UTCDateTime(arrival["time"])
                if arrival.name == evid0:
                # Do the calculation "forward".
                # This means that the primary (earlier) event is used as the template
                # trace.
                    trX, trY = tr0, trB
                    otX, otY = ot0, otB
                else:
                # Do the calculation "backward".
                # This means that the secondary (later) event is used as the template
                # trace.
                    trX, trY = trB, tr0
                    otX, otY = otB, ot0

                # slice the template trace
                trX = trX.slice(starttime=atX-cfg["tlead_%s" % arrival["iphase"].lower()],
                                endtime  =atX+cfg["tlag_%s" % arrival["iphase"].lower()])

                # error checking
                min_nsamp = (cfg["tlead_%s" % arrival["iphase"].lower()]\
                           + cfg["tlag_%s" % arrival["iphase"].lower()]) * trX.stats.sampling_rate
                if len(trX) < min_nsamp or len(trY) < min_nsamp:
                    continue

                # max shift :: the maximum shift to apply when cross-correlating
                # N         :: the half-width (# of samples) of the template
                # corr      :: the cross-correlation time-series
                # clag      :: the lag of the maximum cross-correlation coefficient
                #              relative to the middle of the test trace
                #          ---------|+++++++++
                #          9876543210123456789
                #     trX: -------XXXXX-------
                #     trY: YYYYYYYYYYYYYYYYYYY
                # _ccmax    :: the maximum cross-correlation coefficient
                # shift     :: the lag of the maximum cross-correlation coefficient
                #              relative to the start of the test trace
                #          |+++++++++
                #          0123456789...
                #     trX: XXXXX--------------
                #     trY: YYYYYYYYYYYYYYYYYYY
                # Do the actual correlation
                max_shift    = int(len(trY)/2)
                N            = int(len(trX)/2) # template half-width
                corr         = op.signal.cross_correlation.correlate(trX, trY, max_shift)
                clag, _ccmax = op.signal.cross_correlation.xcorr_max(corr)
                shift        = max_shift-clag-N # peak cross-correlation lag in samples
                # dot    :: differential origin-time
                # dat    :: differntial arrival-time
                # ddiff  :: double-difference (differential travel-time)
                dot   = (otY-otX)
                dat   = (trY.stats.starttime+trY.stats.delta*shift)-atX
                _dbldiff = dot-dat

                if _ccmax >= cfg["corr_min"]:
                    dbldiff.append(_dbldiff)
                    ccmax.append(_ccmax)

            if len(dbldiff) > 0:
                if arrival["sta"] not in measurements:
                    measurements[arrival["sta"]] = {}
                measurements[arrival["sta"]][arrival["iphase"]] = {"dbldiff": np.average(dbldiff, weights=ccmax),
                                                                   "ccmean" : np.mean(ccmax),
                                                                   "chan"   : arrival["chan"]}
        if len(measurements) >= cfg["ncorr_min"]:
            for sta in sorted(measurements):
                for phase in sorted(measurements[sta]):
                    dbldiff = measurements[sta][phase]["dbldiff"]
                    ccmean  = measurements[sta][phase]["ccmean"]
                    chan    = measurements[sta][phase]["chan"]
                    grpid = "{:d}/{:d}/{:s}".format(evid0, evidB, sta)
                    try:
                        f5out[grpid][phase][:] = (dbldiff, ccmean)
                    except:
                        logger.info(grpid, phase, dbldiff, ccmean)
                        raise
        logger.info("correlated event ID#{:d} with ID#{:d} - elapsed time: "\
              "{:.2f} s".format(evid0, evidB, time.time()-log_tstart))

def detect_python_version():
    if sys.version_info.major != 2:
        logger.error("Python2 is currently the only supported version of this"
                     "code. Please use a Python2 interpreter.")
        exit()

if __name__ == "__main__":
    args = parse_args()
    cfg = parse_config(args.config_file)
    configure_logging(args.verbose, args.logfile)
    detect_python_version()
    try:
        main(args, cfg)
    except:
        logger.error(traceback.print_tb(sys.exc_info()[2]))


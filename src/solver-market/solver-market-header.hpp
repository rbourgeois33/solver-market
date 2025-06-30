#include <Kokkos_Core.hpp>

#pragma once 

using Device = Kokkos::DefaultExecutionSpace;
using Host = Kokkos::DefaultHostExecutionSpace;

#ifndef GTEST_ //This is skipped for unit tests as it runs on a simple runner
static_assert(!std::is_same<Device, Host>::value, "Error: Device and Host execution spaces must be different. You build Kokkos from trilinos without GPU option !");
#endif

template<typename _TYPE_>
using HostView=Kokkos::View<_TYPE_*, Host>;

template<typename _TYPE_>
using DeviceView=Kokkos::View<_TYPE_*, Device>;

enum MtxReaderStatus {
    MtxReaderSuccess,
    MtxReaderErrorFileNotFound,
    MtxReaderErrorFileMemAllocFailed,
    MtxReaderErrorWrongNnz,
    MtxReaderErrorUpperViewButLowerFound,
    MtxReaderErrorLowerViewButUpperFound,
    MtxReaderErrorOutOfBoundRowIndex,
    MtxReaderErrorOutOfBoundColIndex,
    MtxReaderUnsupportedObject,
    MtxReaderUnsupportedMatrixType,
    MtxReaderTypeReadIsNotTypeGiven,
    MtxReaderNotAVector,
    MtxReaderWrongHeaderOrNoHeader
};
#!/usr/bin/env python3
"""
Standalone Vizard server — keeps running until you press Ctrl+C.
"""
from Basilisk.utilities import SimulationBaseClass, vizSupport
from Basilisk.simulation import spacecraft

# Minimal sim with one spacecraft (just so the viz module has something to talk to)
scSim = SimulationBaseClass.SimBaseClass()
simTaskName = "simTask"
dynProcess = scSim.CreateNewProcess("simProcess", 10)
dynProcess.addTask(scSim.CreateNewTask(simTaskName, int(1e9)))  # 1 s step

scObject = spacecraft.Spacecraft()
scObject.ModelTag = "bskSat"
scSim.AddModelToTask(simTaskName, scObject)

# Start Vizard server — this will block until Vizard connects
viz = vizSupport.enableUnityVisualization(
    scSim, simTaskName, scObject, liveStream=True
)
viz.reqComProtocol = "tcp"
viz.reqComAddress = "0.0.0.0"
viz.reqPortNumber = "5556"

# No execution loop — just let viz module wait forever in InitializeSimulation
scSim.InitializeSimulation()  # This will block and print "Waiting for Vizard..."
run("Z Project...", "projection=[Sum Slices]");
run("Subtract Background...", "rolling=3 light create sliding");
run("mpl-magma");
//run("Brightness/Contrast...");
resetMinAndMax();
run("Enhance Contrast", "saturated=0.35");


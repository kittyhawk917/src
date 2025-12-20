#!/usr/bin/env nextflow

// The nextflow script containing two processes: scatter and gather. 
// The workflow will kick off the scatter process three times, each with an unique id. 
// The scatter process will generate a random integer between 1 to 100. 
// The gather process will take the output of scatter process and sort them.

// nextflow run scatter_gather.nf -ansi-log false

nextflow.enable.dsl=2

process scatter {
    tag "$id"

    input:
    val id

    output:
    stdout emit: number

    script:
    """
    # Generate random number between 1 and 100
    echo \$(( ( RANDOM % 100 )  + 1 ))
    """
}

process gather {
    debug true

    input:
    val numbers

    script:
    // Sort the collected list of numbers numerically
    def sorted_numbers = numbers.collect { it.trim().toInteger() }.sort()
    """
    echo "Gathered and sorted numbers: ${sorted_numbers.join(', ')}"
    """
}

workflow {
    // 1. Create a channel with three unique IDs to trigger scatter
    ids_ch = Channel.of( 'A', 'B', 'C' )

    // 2. Scatter: Run process for each ID
    scatter(ids_ch)

    // 3. Gather: Collect all outputs into a list and pass to gather
    gather(scatter.out.number.collect())
}


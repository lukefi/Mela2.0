*** Settings ***
Library    Collections
Library    OperatingSystem
Library    Process

*** Keywords ***
Run Simulation
    [Documentation]    Run Mela2.0 simulation with given parameters.
    [Arguments]        ${input_path}    ${output_path}    ${control_path}

    Remove Directory    ${output_path}    recursive=True
    Create Directory    ${output_path}

    ${orig_env}=    Get Environment Variables
    Set To Dictionary    ${orig_env}    PYTHONPATH=${EXECDIR}
    ${result}=    Run Process    python
    ...           -m
    ...           lukefi.metsi.app.metsi
    ...           ${input_path}
    ...           ${output_path}
    ...           ${control_path}
    ...           shell=True
    ...           env=${orig_env}

    Log    STDOUT:\n${result.stdout}
    Log    STDERR:\n${result.stderr}

    Should Be Equal As Integers    ${result.rc}    0    msg=Python script failed! See STDERR log for details.

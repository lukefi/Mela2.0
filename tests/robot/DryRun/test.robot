*** Settings ***
Library    OperatingSystem
Library    Process

*** Test Cases ***
mela2 Command With Help Option Should Complete Successfully
    [Tags]         smoke
    
    ${env} =       Get Environment Variables
    ${result} =    Run Process    python
    ...            -m
    ...            lukefi.mela2.app.mela2
    ...            -h
    ...            shell=True

    Log    STDOUT:\n${result.stdout}
    Log    STDERR:\n${result.stderr}

    Should Be Equal As Integers    ${result.rc}    0
    ...    msg=Python script failed! See STDERR log for details.
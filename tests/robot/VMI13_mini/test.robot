*** Settings ***
Library           ${CURDIR}/../CustomCompareLibrary.py
Resource          ../simulation.resource
Suite Setup       Run Simulation    ${INPUT_PATH}    ${OUTPUT_PATH}    ${CONTROL_SCRIPT}

*** Variables ***
${INPUT_PATH}            ${CURDIR}/input/VMI13_source_mini.dat
${OUTPUT_PATH}           ${CURDIR}/output/test
${CONTROL_SCRIPT}        ${CURDIR}/input/control.py
${REFERENCE_DIR}         ${CURDIR}/output/ref
${ABSOLUTE_TOLERANCE}    0.0000001
${RELATIVE_TOLERANCE}    1e-4

*** Test Cases ***
Simulation State Output Files Should Match Reference
    [Tags]    simulation

    Log To Console    Simulation Succeeded. Verifying output files...

    ${files}=    List Directory Recursively   ${REFERENCE_DIR}
    FOR    ${file}    IN    @{files}
        ${test_file}=        Set Variable    ${OUTPUT_PATH}/${file}
        ${ref_file}=         Set Variable    ${REFERENCE_DIR}/${file}
        File Should Exist    ${test_file}

        Compare Files With Numeric In Text
        ...    ${test_file}
        ...    ${ref_file}
        ...    ${ABSOLUTE_TOLERANCE}
        ...    ${RELATIVE_TOLERANCE}
    END

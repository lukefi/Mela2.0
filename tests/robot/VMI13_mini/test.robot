*** Settings ***
Library           ${CURDIR}/../CustomCompareLibrary.py
Resource          ../simulation.resource

*** Variables ***
${SCRIPT}           -m
${MODULE}           lukefi.metsi.app.metsi
${INPUT_JSON}       ${CURDIR}/input/VMI13_source_mini.dat
${OUTPUT_DIR}       ${CURDIR}/output/test
${CONTROL_SCRIPT}   ${CURDIR}/input/control.py
${REFERENCE_DIR}    ${CURDIR}/output/ref
${TOLERANCE}        0.0000001  # Set your desired tolerance here
${REL_TOL}          1e-4

*** Test Cases ***
Run Simulation And Compare Output Files
    [Tags]    simulation

    Run Simulation    ${INPUT_JSON}    ${OUTPUT_DIR}    ${CONTROL_SCRIPT}

    Log To Console    Simulation Succeeded. Verifying output files...

    ${files}=    List Directory Recursively   ${REFERENCE_DIR}
    FOR    ${file}    IN    @{files}
        ${test_file}=    Set Variable    ${OUTPUT_DIR}/${file}
        ${ref_file}=     Set Variable    ${REFERENCE_DIR}/${file}
        File Should Exist    ${test_file}

        # MODIFIED: Replace the slow keyword with a single call to our fast one.
        # Robot Framework automatically converts the Python function name
        # 'compare_numeric_files_with_tolerance' into this keyword name.
        Compare Files With Numeric In Text    ${test_file}    ${ref_file}    ${TOLERANCE}    ${REL_TOL}
    END

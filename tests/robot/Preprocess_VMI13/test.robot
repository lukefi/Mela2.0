*** Settings ***
Library        ../CustomCompareLibrary.py
Resource       ../simulation.resource
Suite Setup    Run Simulation    ${INPUT_PATH}    ${OUTPUT_PATH}    ${CONTROL_PATH}

*** Variables ***
${INPUT_PATH}            ${CURDIR}/input/VMI13_source_mini.dat
${OUTPUT_PATH}           ${CURDIR}/output/test
${CONTROL_PATH}          ${CURDIR}/input/control.py
${REFERENCE_PATH}        ${CURDIR}/output/ref
${ABSOLUTE_TOLERANCE}    1e-12
${RELATIVE_TOLERANCE}    1e-12

*** Test Cases ***
Preprocessed VMI13 Data Exported As CSV Should Match Reference
    Compare Files With Numeric In Text
    ...    ${REFERENCE_PATH}/preprocessing_result.csv
    ...    ${OUTPUT_PATH}/preprocessing_result.csv
    ...    ${ABSOLUTE_TOLERANCE}
    ...    ${RELATIVE_TOLERANCE}

Preprocessed VMI13 Data Exported As RST Should Match Reference
    Compare Files With Numeric In Text
    ...    ${REFERENCE_PATH}/preprocessing_result.rst
    ...    ${OUTPUT_PATH}/preprocessing_result.rst
    ...    ${ABSOLUTE_TOLERANCE}
    ...    ${RELATIVE_TOLERANCE}
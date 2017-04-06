# 1. Run the program and generate the exodus output

message("Running the command:")
message("${TEST_PROG} " " ${TEST_ARGS}")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} ${TEST_ARGS}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Albany didn't run: test failed")
endif()

# 2. Find and run epu if parallel

if(DEFINED MPIMNP AND ${MPIMNP} GREATER 1)

	if (NOT SEACAS_EPU)
		message(FATAL_ERROR "Cannot find epu")
	endif()

	SET(EPU_COMMAND ${SEACAS_EPU} -auto couponT.exo.${MPIMNP}.0)

  message("Running the command:")
  message("${EPU_COMMAND}")

	EXECUTE_PROCESS(COMMAND ${EPU_COMMAND}
		RESULT_VARIABLE HAD_ERROR)

	if(HAD_ERROR)
		message(FATAL_ERROR "epu failed")
	endif()

endif()


# 2. Find and run exodiff

if (NOT SEACAS_EXODIFF)
  message(FATAL_ERROR "Cannot find exodiff")
endif()

if (NOT SEACAS_ALGEBRA)
  message(FATAL_ERROR "Cannot find algebra")
endif()


if(DEFINED MPIMNP AND ${MPIMNP} GREATER 1)
  SET(EXODIFF_TEST ${SEACAS_EXODIFF} -i -m -f ${DATA_DIR}/${TEST_NAME}.exodiff_commands coupon.b1.exo ${DATA_DIR}/${TEST_NAME}.ref.exo)
ELSE()
  SET(EXODIFF_TEST ${SEACAS_EXODIFF} -i -f ${DATA_DIR}/${TEST_NAME}.exodiff_commands coupon.b1.exo ${DATA_DIR}/${TEST_NAME}.ref.exo)
ENDIF()

message("Running the command:")
message("${EXODIFF_TEST}")

EXECUTE_PROCESS(
    COMMAND ${SEACAS_ALGEBRA} couponT.exo coupon.b1.exo
    INPUT_FILE ${DATA_DIR}/alg.in
    OUTPUT_FILE algebra.out
    RESULT_VARIABLE ALG_ERROR)
if(ALG_ERROR)
	message(FATAL_ERROR "Alegra step failed")
endif()

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST}
    OUTPUT_FILE exodiffT.out
    RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

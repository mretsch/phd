SUBROUTINE first_test(image, n, threshold, output)
    INTEGER      :: n
    REAL(KIND=8) :: threshold
    REAL(KIND=8) :: image (n, n)
    REAL(KIND=8) :: output(n, n)
    !f2py intent(in   )                :: image, threshold
    !f2py intent(hide ), depend(image) :: n = shape(image, 0)
    !f2py intent(  out)                :: output

    WRITE(*,*) "Hello from lovely FORTRAN."

    DO j=1,n
        DO i=1,n
            IF (image(i,j) .gt. threshold) THEN
                output(i,j) = 1.0
            ELSE
                output(i,j) = 0.0
            END IF
        END DO
    END DO
END SUBROUTINE first_test

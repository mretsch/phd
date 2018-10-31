SUBROUTINE histogram_2d(x, y, l, n, xbound, ybound, hist_2d, xedges, yedges)
    INTEGER      :: l            ! length of x & y
    INTEGER      :: n            ! number of bins
    REAL(KIND=8) :: x(l)
    REAL(KIND=8) :: y(l)
    REAL(KIND=8) :: xbound(2)    ! min and max boundaries for bin selection
    REAL(KIND=8) :: ybound(2)    ! min and max boundaries for bin selection
    REAL(KIND=8) :: hist_2d(n, n)
    REAL(KIND=8) :: xedges(n+1)
    REAL(KIND=8) :: yedges(n+1)
    !f2py intent(in   )            :: n, x, y, xbound, ybound
    !f2py intent(hide ), depend(x) :: l = shape(x, 1)
    !f2py intent(  out)            :: xedges, yedges, hist_2d

    REAL(KIND=8) :: xstep, ystep, ymasked(l)
    LOGICAL      :: l_mask(l)

    xedges(  1) = xbound(1)
    xedges(n+1) = xbound(2)
    yedges(  1) = ybound(1)
    yedges(n+1) = ybound(2)

    xstep = ABS(xbound(1) - xbound(2)) / n
    ystep = ABS(ybound(1) - ybound(2)) / n
    DO i=2,n
        xedges(i) = xedges(i-1) + xstep
        yedges(i) = yedges(i-1) + ystep
    END DO

    hist_2d = 0.

    DO i=1,n-1
        ! mask y to bins of x
        !l_mask = False
        !ymasked = ybound(1)-1
        !WHERE ((xedges(i) .LE. x) .AND. (x .LT. xedges(i+1))) l_mask = True
        !ymasked = MERGE(y, ymasked, l_mask)
        DO k=1,l
            ymasked(k) = MERGE(y(k), ybound(1)-1, (xedges(i) .LE. x(k)) .AND. (x(k) .LT. xedges(i+1)))
        END DO

        ! Count ymasked
        DO j=1,n-1
            DO m=1,l
                hist_2d(j,i) = MERGE(hist_2d(j,i) + 1,                                              &
                                     hist_2d(j,i)    ,                                              &
                                     (yedges(j) .LE. ymasked(m)) .AND. (ymasked(m) .LT. yedges(j+1)))
            END DO
        END DO
        ! for the last y-bin take all data (LT --> LE)
        DO m=1,l
            hist_2d(n,i) = MERGE(hist_2d(n,i) + 1,                                              &
                                 hist_2d(n,i)    ,                                              &
                                 (yedges(n) .LE. ymasked(m)) .AND. (ymasked(m) .LE. yedges(n+1)))
        END DO
    END DO

    ! for the last x-bin take all data (LT --> LE)
    !l_mask = False
    !ymasked = ybound(1)-1
    !WHERE ((xedges(n) .LE. x) .AND. (x .LE. xedges(n+1))) l_mask = True
    !ymasked = MERGE(y, ymasked, l_mask)
    DO k=1,l
        ymasked(k) = MERGE(y(k), ybound(1)-1, (xedges(n) .LE. x(k)) .AND. (x(k) .LE. xedges(n+1)))
    END DO
    ! Count ymasked
    DO j=1,n-1
        DO m=1,l
            hist_2d(j,n) = MERGE(hist_2d(j,n) + 1,                                              &
                                 hist_2d(j,n)    ,                                              &
                                 (yedges(j) .LE. ymasked(m)) .AND. (ymasked(m) .LT. yedges(j+1)))
        END DO
    END DO
    ! for the last y-bin take all data (LT --> LE)
    DO m=1,l
        hist_2d(n,n) = MERGE(hist_2d(n,n) + 1,                                              &
                             hist_2d(n,n)    ,                                              &
                             (yedges(n) .LE. ymasked(m)) .AND. (ymasked(m) .LE. yedges(n+1)))
    END DO

    WRITE(*,*) "Hello from lovely FORTRAN."

END SUBROUTINE histogram_2d

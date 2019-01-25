SUBROUTINE histogram_2d(xseries, yseries, length, xedges, yedges, l_cut_off, cut_off, &
                        hist_2d, xwhichbins, ywhichbins)

    INTEGER      :: length                !     length of xseries & yseries
    INTEGER      :: xnbins                !     number of bins for x-axis
    INTEGER      :: ynbins                !     number of bins for y-axis
    INTEGER      :: hist_2d(nbins, nbins) ! OUT
    INTEGER      :: cut_off               !  IN pixels containing less than cut_off cases will not returned

    REAL(KIND=8) :: xseries(length)       !  IN
    REAL(KIND=8) :: yseries(length)       !  IN
    REAL(KIND=8) :: xedges(xnbins+1)      !  IN
    REAL(KIND=8) :: yedges(ynbins+1)      !  IN
    REAL(KIND=8) :: xwhichbins(length)    ! OUT
    REAL(KIND=8) :: ywhichbins(length)    ! OUT

    LOGICAL      :: l_cut_off             !  IN

    !f2py intent(in   )                  :: xseries, yseries, xedges, yedges, l_cut_off, cut_off
    !f2py intent(hide ), depend(xseries) :: length=shape(xseries,1)
    !f2py intent(hide ), depend(xedges)  :: xnbins=shape(xedges,1)-1
    !f2py intent(hide ), depend(yedges)  :: ynbins=shape(yedges,1)-1
    !f2py intent(  out)                  :: hist_2d, xwhichbins, ywhichbins

    INTEGER      :: x_indx, y_indx
    REAL(KIND=8) :: ymasked(length), xbins(nbins), ybins(nbins), tmp(nbins)
    LOGICAL      :: l_mask(length), l_temp

    DO n=1,nbins
        xbins(n) = xedges(n) + (xedges(n+1) - xedges(n)) / 2.
        ybins(n) = yedges(n) + (yedges(n+1) - yedges(n)) / 2.
    END DO

    hist_2d = 0
    xwhichbins = -1.
    ywhichbins = -1.

    DO i=1,nbins-1
        ! mask yseries to bins of xseries
        !l_mask = False
        !ymasked = ybound(1)-1
        !WHERE ((xedges(i) .LE. xseries) .AND. (x .LT. xedges(i+1))) l_mask = True
        !ymasked = MERGE(yseries, ymasked, l_mask)
        DO k=1,length
            ymasked(k) = MERGE(yseries(k), ybound(1)-1, (xedges(i) .LE. xseries(k)) .AND. (xseries(k) .LT. xedges(i+1)))
        END DO
        ! save which (time) step fell into which bin
        xwhichbins = MERGE(xbins(i), xwhichbins, ymasked .NE. ybound(1)-1)

        ! Count ymasked
        DO j=1,nbins-1
            DO m=1,length
                l_temp = (yedges(j) .LE. ymasked(m)) .AND. (ymasked(m) .LT. yedges(j+1))
                hist_2d(j,i) = MERGE(hist_2d(j,i) + 1,&
                                     hist_2d(j,i)    ,&
                                     l_temp           )
                ! save which (time) step fell into which bin
                ywhichbins(m) = MERGE(ybins(j), ywhichbins(m), l_temp)
            END DO
        END DO

        ! for the last yseries-bin take all data (LT --> LE)
        DO m=1,length
            l_temp = (yedges(nbins) .LE. ymasked(m)) .AND. (ymasked(m) .LE. yedges(nbins+1))
            hist_2d(nbins,i) = MERGE(hist_2d(nbins,i) + 1,&
                                     hist_2d(nbins,i)    ,&
                                     l_temp               )
            ! save which (time) step fell into which bin
            ywhichbins(m) = MERGE(ybins(nbins), ywhichbins(m), l_temp)
        END DO
    END DO

    ! for the last xseries-bin take all data (LT --> LE)
    DO k=1,length
        ymasked(k) = MERGE(yseries(k), ybound(1)-1, (xedges(nbins) .LE. xseries(k)) .AND. (xseries(k) .LE. xedges(nbins+1)))
    END DO
    ! save which (time) step fell into which bin
    xwhichbins = MERGE(xbins(nbins), xwhichbins, ymasked .NE. ybound(1)-1)
    ! Count ymasked
    DO j=1,nbins-1
        DO m=1,length
            l_temp = (yedges(j) .LE. ymasked(m)) .AND. (ymasked(m) .LT. yedges(j+1))
            hist_2d(j,nbins) = MERGE(hist_2d(j,nbins) + 1,&
                                     hist_2d(j,nbins)    ,&
                                     l_temp               )
            ! save which (time) step fell into which bin
            ywhichbins(m) = MERGE(ybins(j), ywhichbins(m), l_temp)
        END DO
    END DO
    ! for the last yseries-bin take all data (LT --> LE)
    DO m=1,length
        l_temp = (yedges(nbins) .LE. ymasked(m)) .AND. (ymasked(m) .LE. yedges(nbins+1))
        hist_2d(nbins,nbins) = MERGE(hist_2d(nbins,nbins) + 1,&
                                     hist_2d(nbins,nbins)    ,&
                                     l_temp                   )
        ! save which (time) step fell into which bin
        ywhichbins(m) = MERGE(ybins(nbins), ywhichbins(m), l_temp)
    END DO

    ! Get rid of (time) steps which contributed only to weakly populated pixels
    IF (l_cut_off) THEN
        DO m=1,length
            !x_indx = FINDLOC(xbins, xwhichbins(m), 1)
            !y_indx = FINDLOC(ybins, ywhichbins(m), 1)
            tmp = ABS(ybins - ywhichbins(m))
            y_indx = MINLOC(tmp, 1)
            tmp = ABS(xbins - xwhichbins(m))
            x_indx = MINLOC(tmp, 1)
            IF (hist_2d(y_indx, x_indx) .LT. cut_off) THEN
                ywhichbins(m) = -1.
                xwhichbins(m) = -1.
            END IF
        END DO
        DO i=1,nbins
            DO j=1,nbins
                IF (hist_2d(j,i) .LT. cut_off) THEN
                    hist_2d(j,i) = 0
                END IF
            END DO
        END DO
    END IF

    WRITE(*,*) "Hello from lovely FORTRAN."

END SUBROUTINE histogram_2d

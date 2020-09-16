SUBROUTINE histogram_2d(xseries, yseries, length, &
                        xedges, yedges, nxbins, nybins, &
                        l_cut_off, cut_off, l_density, &
                        hist_2d, xwhichbins, ywhichbins)

    INTEGER      :: length                  !     length of xseries & yseries
    INTEGER      :: nxbins                  !     number of bins for x-axis
    INTEGER      :: nybins                  !     number of bins for y-axis
    INTEGER      :: cut_off                 !  IN pixels containing less than cut_off cases will not returned

    REAL(KIND=8) :: xseries(length)         !  IN
    REAL(KIND=8) :: yseries(length)         !  IN
    REAL(KIND=8) :: xedges(nxbins+1)        !  IN
    REAL(KIND=8) :: yedges(nybins+1)        !  IN
    REAL(KIND=8) :: hist_2d(nybins, nxbins) ! OUT
    REAL(KIND=8) :: xwhichbins(length)      ! OUT
    REAL(KIND=8) :: ywhichbins(length)      ! OUT

    LOGICAL      :: l_cut_off               !  IN
    LOGICAL      :: l_density               !  IN

    !f2py intent(in   )                  :: xseries, yseries, xedges, yedges, l_cut_off, cut_off, l_density
    !f2py intent(hide ), depend(xseries) :: length=shape(xseries,1)
    !f2py intent(hide ), depend(xedges)  :: nxbins=shape(xedges,1)-1
    !f2py intent(hide ), depend(yedges)  :: nybins=shape(yedges,1)-1
    !f2py intent(  out)                  :: xwhichbins, ywhichbins, hist_2d

    INTEGER      :: x_indx, y_indx
    REAL(KIND=8) :: ymasked(length), xbins(nxbins), ybins(nybins), dx(nxbins), dy(nybins), hist_sum
    REAL(KIND=8) :: tmp_x(nxbins), tmp_y(nybins), fillvalue
    LOGICAL      :: l_mask(length), l_temp

    DO n=1,nxbins
        xbins(n) = xedges(n) + (xedges(n+1) - xedges(n)) / 2.
        dx(n)    = xedges(n+1) - xedges(n)
    END DO
    DO n=1,nybins
        ybins(n) = yedges(n) + (yedges(n+1) - yedges(n)) / 2.
        dy(n)    = yedges(n+1) - yedges(n)
    END DO

    fillvalue  = -999999999.
    xwhichbins = -1.
    ywhichbins = -1.
    hist_2d = 0.

    DO i=1,nxbins-1
        ! mask yseries to bins of xseries
        !l_mask = False
        !ymasked = ybound(1)-1
        !WHERE ((xedges(i) .LE. xseries) .AND. (x .LT. xedges(i+1))) l_mask = True
        !ymasked = MERGE(yseries, ymasked, l_mask)
        DO k=1,length
            ymasked(k) = MERGE(yseries(k), fillvalue, (xedges(i) .LE. xseries(k)) .AND. (xseries(k) .LT. xedges(i+1)))
        END DO
        ! save which (time) step fell into which bin
        xwhichbins = MERGE(xbins(i), xwhichbins, ymasked .NE. fillvalue)

        ! Count ymasked
        DO j=1,nybins-1
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
            l_temp = (yedges(nybins) .LE. ymasked(m)) .AND. (ymasked(m) .LE. yedges(nybins+1))
            hist_2d(nybins,i) = MERGE(hist_2d(nybins,i) + 1,&
                                      hist_2d(nybins,i)    ,&
                                      l_temp                )
            ! save which (time) step fell into which bin
            ywhichbins(m) = MERGE(ybins(nybins), ywhichbins(m), l_temp)
        END DO
    END DO

    ! for the last xseries-bin take all data (LT --> LE)
    DO k=1,length
        ymasked(k) = MERGE(yseries(k), fillvalue, (xedges(nxbins) .LE. xseries(k)) .AND. (xseries(k) .LE. xedges(nxbins+1)))
    END DO
    ! save which (time) step fell into which bin
    xwhichbins = MERGE(xbins(nxbins), xwhichbins, ymasked .NE. fillvalue)
    ! Count ymasked
    DO j=1,nybins-1
        DO m=1,length
            l_temp = (yedges(j) .LE. ymasked(m)) .AND. (ymasked(m) .LT. yedges(j+1))
            hist_2d(j,nxbins) = MERGE(hist_2d(j,nxbins) + 1,&
                                      hist_2d(j,nxbins)    ,&
                                      l_temp                )
            ! save which (time) step fell into which bin
            ywhichbins(m) = MERGE(ybins(j), ywhichbins(m), l_temp)
        END DO
    END DO
    ! for the last yseries-bin take all data (LT --> LE)
    DO m=1,length
        l_temp = (yedges(nybins) .LE. ymasked(m)) .AND. (ymasked(m) .LE. yedges(nybins+1))
        hist_2d(nybins,nxbins) = MERGE(hist_2d(nybins,nxbins) + 1,&
                                       hist_2d(nybins,nxbins)    ,&
                                       l_temp                     )
        ! save which (time) step fell into which bin
        ywhichbins(m) = MERGE(ybins(nybins), ywhichbins(m), l_temp)
    END DO

    ! Get rid of (time) steps which contributed only to weakly populated pixels
    IF (l_cut_off) THEN
        DO m=1,length
            !x_indx = FINDLOC(xbins, xwhichbins(m), 1)
            !y_indx = FINDLOC(ybins, ywhichbins(m), 1)
            tmp_y  = ABS(ybins - ywhichbins(m))
            y_indx = MINLOC(tmp_y, 1)
            tmp_x  = ABS(xbins - xwhichbins(m))
            x_indx = MINLOC(tmp_x, 1)
            IF (hist_2d(y_indx, x_indx) .LT. cut_off) THEN
                ywhichbins(m) = -1.
                xwhichbins(m) = -1.
            END IF
        END DO
        DO i=1,nxbins
            DO j=1,nybins
                IF (hist_2d(j,i) .LT. cut_off) THEN
                    hist_2d(j,i) = 0
                END IF
            END DO
        END DO
    END IF

    ! probability density
    IF (l_density) THEN
        hist_sum = sum(hist_2d)
        DO i=1,nxbins
            DO j=1,nybins
                hist_2d(j,i) = hist_2d(j,i) / (dx(i) * dy(j) * hist_sum)
            END DO
        END DO
    END IF


    WRITE(*,*) "Hello from lovely FORTRAN."

END SUBROUTINE histogram_2d

SUBROUTINE phasespace(indices1, indices2, mn, overlay, overlay_x, overlay_y, length, &
                      l_probability, upper_bound, lower_bound, bin_values)
    USE ieee_arithmetic

    INTEGER      :: mn                   ! dimension size of flattened 2d-histogram
    INTEGER      :: length               ! length of the time series to put 'into' 2d-histogram

    REAL(KIND=8) :: indices1(mn)         !  IN
    REAL(KIND=8) :: indices2(mn)         !  IN
    REAL(KIND=8) :: overlay  (length)    !  IN
    REAL(KIND=8) :: overlay_x(length)    !  IN
    REAL(KIND=8) :: overlay_y(length)    !  IN
    REAL(KIND=8) :: upper_bound          !  IN
    REAL(KIND=8) :: lower_bound          !  IN
    REAL(KIND=8) :: bin_values(mn)       !  OUT

    LOGICAL      :: l_probability        !  IN

    !f2py intent(in  )                   :: indices1, indices2, overlay, overlay_x, overlay_y
    !f2py intent(in  )                   :: upper_bound, lower_bound, l_probability
    !f2py intent(hide), depend(indices1) :: mn
    !f2py intent(hide), depend(overlay ) :: length
    !f2py intent( out)                   :: bin_values

    LOGICAL      :: l_1(length), l_2(length), l_12(length), l_hit_nans(length), l_in_bounds(length)
    INTEGER      :: i_12(length), total(mn), i_in_bounds(length), n_in_bounds(mn)
    REAL(KIND=8) :: hit_values(length), hit_value_sum(mn)

    DO i=1, mn
        l_1 = overlay_x .EQ. indices1(i)
        l_2 = overlay_y .EQ. indices2(i)
        l_12 = l_1 .AND. l_2

        ! subselect values of the overlay time series for one histogram-bin, set rest to zero
        hit_values = MERGE(overlay, 0.d0, l_12)
        ! if hit_values contain NaNs, also set them to zero
        l_hit_nans = IEEE_IS_NAN(hit_values)
        hit_values = MERGE(0.d0, hit_values, l_hit_nans)
        ! integer converted from logical of indices that survived filtering
        i_12 = l_12 .AND. (.NOT. l_hit_nans)

        IF (l_probability) THEN
            ! check for values to be inside boundaries
            l_in_bounds = (lower_bound .LE. hit_values) .AND. (hit_values .LE. upper_bound)
            i_in_bounds = l_12 .AND. (l_in_bounds)

            ! total number of values inside the boundaries
            n_in_bounds(i) = sum(i_in_bounds)
        END IF

        total(i)         = sum(i_12)
        hit_value_sum(i) = sum(hit_values)
    END DO

    IF (l_probability) THEN
        ! compute the percentage of values inside bounds to number of survived values
        ! Fortran-NaNs are converted to 0. in python, so choose a special value to check against in python
        bin_values = MERGE(-9999999999.d0, REAL(n_in_bounds, KIND=8) / total, total .EQ. 0)
    ELSE
        ! compute the average value of survived values
        ! Fortran-NaNs are converted to 0. in python, so choose a special value to check against in python
        bin_values = MERGE(-9999999999.d0, hit_value_sum / total, total .EQ. 0)
    END IF

    WRITE(*,*) "Hello from lovely FORTRAN again."
END SUBROUTINE phasespace

// ============================================================================
// Combined self + distinct MSD calculator for a THREE-SPECIES system
// (cation, anion1, anion2 -- no water), WITH BLOCK AVERAGING:
//
//   D+   : cation  self diffusion
//   D1   : anion1  self diffusion
//   D2   : anion2  self diffusion
//   D++  : cation-cation   distinct (correlation between DIFFERENT cations)
//   D11  : anion1-anion1   distinct (correlation between DIFFERENT anion1's)
//   D22  : anion2-anion2   distinct (correlation between DIFFERENT anion2's)
//   D+1  : cation-anion1   distinct
//   D+2  : cation-anion2   distinct
//   D12  : anion1-anion2   distinct 
//   Compile with:
//
//g++ -std=c++17 -O3 -fopenmp conductivity_xtc_standalone.cpp \
//       -I$HOME/xdrfile-1.1.4/include -L$HOME/.local/lib \
//       -Wl,-rpath,$HOME/.local/lib -lxdrfile \
//       -o conductivity_xtc_standalone
// Run with:
//   ./conductivity_xtc_standalone
//
#include<iostream>
#include<fstream>
#include<sstream>
#include<iomanip>
#include<cmath>
#include<string>
#include<vector>
#include<omp.h>
#include "xdrfile.h"
#include "xdrfile_xtc.h"

using namespace std;

int main()
{
    int nstep = 100000;

    int ncation     = 1000;
    int ncationatm  = 10;
    int nanion1     = 250;
    int nanion1atm  = 5;
    int nanion2     = 750;
    int nanion2atm  = 9;

    double cation_x[ncation][ncationatm], cation_y[ncation][ncationatm], cation_z[ncation][ncationatm];
    double anion1_x[nanion1][nanion1atm], anion1_y[nanion1][nanion1atm], anion1_z[nanion1][nanion1atm];
    double anion2_x[nanion2][nanion2atm], anion2_y[nanion2][nanion2atm], anion2_z[nanion2][nanion2atm];

    double com_cation_x[ncation], com_cation_y[ncation], com_cation_z[ncation];
    double com_anion1_x[nanion1], com_anion1_y[nanion1], com_anion1_z[nanion1];
    double com_anion2_x[nanion2], com_anion2_y[nanion2], com_anion2_z[nanion2];

    double L = 0.0;
    double hL = 0.0;
    const double xtc_frame_dt = 2.0;
    const int    xtc_stride   = 1;
    const double timestep     = xtc_frame_dt * xtc_stride;

    vector<vector<double>> cation_com_x_t(nstep, vector<double>(ncation));
    vector<vector<double>> cation_com_y_t(nstep, vector<double>(ncation));
    vector<vector<double>> cation_com_z_t(nstep, vector<double>(ncation));

    vector<vector<double>> anion1_com_x_t(nstep, vector<double>(nanion1));
    vector<vector<double>> anion1_com_y_t(nstep, vector<double>(nanion1));
    vector<vector<double>> anion1_com_z_t(nstep, vector<double>(nanion1));

    vector<vector<double>> anion2_com_x_t(nstep, vector<double>(nanion2));
    vector<vector<double>> anion2_com_y_t(nstep, vector<double>(nanion2));
    vector<vector<double>> anion2_com_z_t(nstep, vector<double>(nanion2));

    vector<double> system_com_x_t(nstep), system_com_y_t(nstep), system_com_z_t(nstep);

    const double m_cation[ncationatm] = {12.0110, 14.0070, 14.0070, 14.0070, 1.0080, 1.0080, 1.0080, 1.0080, 1.0080, 1.0080};
    const double m_anion1[nanion1atm] = {26.982, 35.453, 35.453, 35.453, 35.453};
    const double m_anion2[nanion2atm] = {26.982, 35.453, 35.453, 26.982, 35.453, 35.453, 35.453, 35.453, 35.453};

    double total_mass_cation = 0.0, total_mass_anion1 = 0.0, total_mass_anion2 = 0.0;
    for (int i = 0; i < ncationatm; i++) total_mass_cation += m_cation[i];
    for (int i = 0; i < nanion1atm; i++) total_mass_anion1 += m_anion1[i];
    for (int i = 0; i < nanion2atm; i++) total_mass_anion2 += m_anion2[i];

    double total_mass_system = (ncation*total_mass_cation) + (nanion1*total_mass_anion1) + (nanion2*total_mass_anion2);

    // ========================================================================
    // Reading XTC file
    // ========================================================================

    char xtc_filename[] = "run.xtc";
    XDRFILE *xtc = xdrfile_open(xtc_filename, "r");
    if (!xtc)
    {
        cerr << "ERROR: Could not open run.xtc" << endl;
        return 1;
    }

    int xtc_natoms = 0;
    if (read_xtc_natoms(xtc_filename, &xtc_natoms) != exdrOK)
    {
        cerr << "ERROR: Could not read atom count from run.xtc" << endl;
        xdrfile_close(xtc);
        return 1;
    }

    const int expected_natoms = ncation*ncationatm + nanion1*nanion1atm + nanion2*nanion2atm;

    if (xtc_natoms != expected_natoms)
    {
        cerr << "ERROR: run.xtc contains " << xtc_natoms
             << " atoms, but this analysis expects " << expected_natoms
             << " atoms." << endl;
        xdrfile_close(xtc);
        return 1;
    }

    vector<rvec> xtc_x(xtc_natoms);
    matrix box;
    float xtc_time = 0.0f;
    float xtc_prec = 0.0f;
    int xtc_step = 0;

    const long long total_xtc_frames_needed = 1LL * nstep * xtc_stride;
    long long xtc_global_frame = 0;
    double previous_xtc_time = 0.0;

    auto check_box = [&](long long frame_number) -> bool
    {
        double Lx = double(box[0][0]) * 10.0;
        double Ly = double(box[1][1]) * 10.0;
        double Lz = double(box[2][2]) * 10.0;

        const double tol = 1.0e-4;
        if (fabs(Lx - Ly) > tol || fabs(Lx - Lz) > tol)
        {
            cerr << "ERROR: run.xtc box is not cubic at frame " << frame_number
                 << ": " << Lx << " x " << Ly << " x " << Lz
                 << " Angstrom." << endl;
            cerr << "This version preserves the original single-L PBC treatment." << endl;
            return false;
        }

        if (frame_number == 0)
        {
            L = Lx;
            hL = L/2.0;
            double vol_A3 = L * L * L;

    	    cout << "XTC atom count          : " << xtc_natoms << endl;
    	    cout << "XTC box (frame 0)       : " << Lx << " x " << Ly << " x " << Lz << " Angstrom" << endl;
    	    cout << "XTC Volume (NVT)        : " << setprecision(10) << vol_A3 << " Angstrom^3 (" << vol_A3 / 1000.0 << " nm^3)" << endl;
            cout << "XTC native frame dt     : " << xtc_frame_dt << " ps" << endl;
            cout << "Analysis frame dt       : " << timestep << " ps" << endl;
            cout << "XTC frame stride        : " << xtc_stride << endl;
            cout << "Selected analysis frames: " << nstep << endl;
            cout << "Native XTC frames read  : " << total_xtc_frames_needed << endl;
            cout << "XTC first time          : " << xtc_time << " ps" << endl;
        }
        return true;
    };

    auto read_one_xtc_frame = [&](long long frame_number) -> bool
    {
        int status = read_xtc(xtc, xtc_natoms, &xtc_step, &xtc_time,
                              box, xtc_x.data(), &xtc_prec);
        if (status != exdrOK)
        {
            cerr << "ERROR: Could not read XTC frame " << frame_number
                 << ". Successfully read " << frame_number << " frames." << endl;
            return false;
        }
        if (!check_box(frame_number))
            return false;
        return true;
    };

    for (int i = 0; i < nstep; i++)
    {
        if (!read_one_xtc_frame(xtc_global_frame))
        {
            xdrfile_close(xtc);
            return 1;
        }

        if (i == 0)
        {
            previous_xtc_time = double(xtc_time);
        }

        if (i % 100 == 0)
            cout << "Analyzing frame " << i
                 << " (XTC frame " << xtc_global_frame
                 << ", step " << xtc_step << ", time " << xtc_time << " ps)" << endl;

        // ===== Cation COM and PBC Minimum image convention 
        int atom_index = 0;
        for (int j = 0; j < ncation; j++)
        {
            for (int k = 0; k < ncationatm; k++, atom_index++)
            {
                cation_x[j][k] = double(xtc_x[atom_index][0]) * 10.0;
                cation_y[j][k] = double(xtc_x[atom_index][1]) * 10.0;
                cation_z[j][k] = double(xtc_x[atom_index][2]) * 10.0;
            }

            double refx = cation_x[j][0], refy = cation_y[j][0], refz = cation_z[j][0];
            for (int k = 1; k < ncationatm; k++)
            {
                double dx = cation_x[j][k] - refx;
                double dy = cation_y[j][k] - refy;
                double dz = cation_z[j][k] - refz;
                if (dx > hL) dx -= L; else if (dx <= -hL) dx += L;
                if (dy > hL) dy -= L; else if (dy <= -hL) dy += L;
                if (dz > hL) dz -= L; else if (dz <= -hL) dz += L;
                cation_x[j][k] = refx + dx;
                cation_y[j][k] = refy + dy;
                cation_z[j][k] = refz + dz;
            }

            com_cation_x[j] = com_cation_y[j] = com_cation_z[j] = 0.0;
            for (int k = 0; k < ncationatm; k++)
            {
                com_cation_x[j] += cation_x[j][k] * m_cation[k];
                com_cation_y[j] += cation_y[j][k] * m_cation[k];
                com_cation_z[j] += cation_z[j][k] * m_cation[k];
            }
            com_cation_x[j] /= total_mass_cation;
            com_cation_y[j] /= total_mass_cation;
            com_cation_z[j] /= total_mass_cation;

            cation_com_x_t[i][j] = com_cation_x[j];
            cation_com_y_t[i][j] = com_cation_y[j];
            cation_com_z_t[i][j] = com_cation_z[j];
        }

        // ===== Anion1 COM and PBC Minimum image convention
        for (int j = 0; j < nanion1; j++)
        {
            for (int k = 0; k < nanion1atm; k++, atom_index++)
            {
                anion1_x[j][k] = double(xtc_x[atom_index][0]) * 10.0;
                anion1_y[j][k] = double(xtc_x[atom_index][1]) * 10.0;
                anion1_z[j][k] = double(xtc_x[atom_index][2]) * 10.0;
            }

            double refx = anion1_x[j][0], refy = anion1_y[j][0], refz = anion1_z[j][0];
            for (int k = 1; k < nanion1atm; k++)
            {
                double dx = anion1_x[j][k] - refx;
                double dy = anion1_y[j][k] - refy;
                double dz = anion1_z[j][k] - refz;
                if (dx > hL) dx -= L; else if (dx <= -hL) dx += L;
                if (dy > hL) dy -= L; else if (dy <= -hL) dy += L;
                if (dz > hL) dz -= L; else if (dz <= -hL) dz += L;
                anion1_x[j][k] = refx + dx;
                anion1_y[j][k] = refy + dy;
                anion1_z[j][k] = refz + dz;
            }

            com_anion1_x[j] = com_anion1_y[j] = com_anion1_z[j] = 0.0;
            for (int k = 0; k < nanion1atm; k++)
            {
                com_anion1_x[j] += anion1_x[j][k] * m_anion1[k];
                com_anion1_y[j] += anion1_y[j][k] * m_anion1[k];
                com_anion1_z[j] += anion1_z[j][k] * m_anion1[k];
            }
            com_anion1_x[j] /= total_mass_anion1;
            com_anion1_y[j] /= total_mass_anion1;
            com_anion1_z[j] /= total_mass_anion1;

            anion1_com_x_t[i][j] = com_anion1_x[j];
            anion1_com_y_t[i][j] = com_anion1_y[j];
            anion1_com_z_t[i][j] = com_anion1_z[j];
        }

        // ===== Anion2 COM and PBC Minimum image convention
        for (int j = 0; j < nanion2; j++)
        {
            for (int k = 0; k < nanion2atm; k++, atom_index++)
            {
                anion2_x[j][k] = double(xtc_x[atom_index][0]) * 10.0;
                anion2_y[j][k] = double(xtc_x[atom_index][1]) * 10.0;
                anion2_z[j][k] = double(xtc_x[atom_index][2]) * 10.0;
            }

            double refx = anion2_x[j][0], refy = anion2_y[j][0], refz = anion2_z[j][0];
            for (int k = 1; k < nanion2atm; k++)
            {
                double dx = anion2_x[j][k] - refx;
                double dy = anion2_y[j][k] - refy;
                double dz = anion2_z[j][k] - refz;
                if (dx > hL) dx -= L; else if (dx <= -hL) dx += L;
                if (dy > hL) dy -= L; else if (dy <= -hL) dy += L;
                if (dz > hL) dz -= L; else if (dz <= -hL) dz += L;
                anion2_x[j][k] = refx + dx;
                anion2_y[j][k] = refy + dy;
                anion2_z[j][k] = refz + dz;
            }

            com_anion2_x[j] = com_anion2_y[j] = com_anion2_z[j] = 0.0;
            for (int k = 0; k < nanion2atm; k++)
            {
                com_anion2_x[j] += anion2_x[j][k] * m_anion2[k];
                com_anion2_y[j] += anion2_y[j][k] * m_anion2[k];
                com_anion2_z[j] += anion2_z[j][k] * m_anion2[k];
            }
            com_anion2_x[j] /= total_mass_anion2;
            com_anion2_y[j] /= total_mass_anion2;
            com_anion2_z[j] /= total_mass_anion2;

            anion2_com_x_t[i][j] = com_anion2_x[j];
            anion2_com_y_t[i][j] = com_anion2_y[j];
            anion2_com_z_t[i][j] = com_anion2_z[j];
        }

        // ===== SYSTEM COM FOR THIS FRAME =====
        double sysx = 0.0, sysy = 0.0, sysz = 0.0;
        for (int j = 0; j < ncation; j++)
        {
            sysx += com_cation_x[j] * total_mass_cation;
            sysy += com_cation_y[j] * total_mass_cation;
            sysz += com_cation_z[j] * total_mass_cation;
        }
        for (int j = 0; j < nanion1; j++)
        {
            sysx += com_anion1_x[j] * total_mass_anion1;
            sysy += com_anion1_y[j] * total_mass_anion1;
            sysz += com_anion1_z[j] * total_mass_anion1;
        }
        for (int j = 0; j < nanion2; j++)
        {
            sysx += com_anion2_x[j] * total_mass_anion2;
            sysy += com_anion2_y[j] * total_mass_anion2;
            sysz += com_anion2_z[j] * total_mass_anion2;
        }
        system_com_x_t[i] = sysx / total_mass_system;
        system_com_y_t[i] = sysy / total_mass_system;
        system_com_z_t[i] = sysz / total_mass_system;

        for (int skip = 1; skip < xtc_stride; skip++)
        {
            ++xtc_global_frame;
            if (!read_one_xtc_frame(xtc_global_frame))
            {
                xdrfile_close(xtc);
                return 1;
            }

            if (xtc_global_frame == 1)
            {
                double observed_dt = double(xtc_time) - previous_xtc_time;
                if (observed_dt <= 0.0)
                {
                    cerr << "ERROR: XTC frame times are not increasing." << endl;
                    xdrfile_close(xtc);
                    return 1;
                }
                if (fabs(observed_dt - xtc_frame_dt) > 1.0e-3)
                {
                    cerr << "ERROR: Expected XTC frame spacing of " << xtc_frame_dt
                         << " ps, but observed " << observed_dt << " ps." << endl;
                    cerr << "The 5-frame stride would not produce the intended 10-ps analysis spacing." << endl;
                    xdrfile_close(xtc);
                    return 1;
                }
                cout << "Verified XTC native dt   : " << observed_dt << " ps" << endl;
            }
        }
        ++xtc_global_frame;
    }
    xdrfile_close(xtc);
    cout << "Finished reading all " << nstep << " frames. Unwrapping..." << endl;

    // ===== TIME-UNWRAP SYSTEM COM =====
    vector<double> sys_unwrapped_x(nstep), sys_unwrapped_y(nstep), sys_unwrapped_z(nstep);
    sys_unwrapped_x[0] = system_com_x_t[0];
    sys_unwrapped_y[0] = system_com_y_t[0];
    sys_unwrapped_z[0] = system_com_z_t[0];
    for (int i = 1; i < nstep; i++)
    {
        double dx = system_com_x_t[i] - system_com_x_t[i-1];
        double dy = system_com_y_t[i] - system_com_y_t[i-1];
        double dz = system_com_z_t[i] - system_com_z_t[i-1];
        if (dx > hL) dx -= L; else if (dx <= -hL) dx += L;
        if (dy > hL) dy -= L; else if (dy <= -hL) dy += L;
        if (dz > hL) dz -= L; else if (dz <= -hL) dz += L;
        sys_unwrapped_x[i] = sys_unwrapped_x[i-1] + dx;
        sys_unwrapped_y[i] = sys_unwrapped_y[i-1] + dy;
        sys_unwrapped_z[i] = sys_unwrapped_z[i-1] + dz;
    }

    // ===== TIME-UNWRAP CATION COM =====
    vector<vector<double>> cation_pos_x(nstep, vector<double>(ncation));
    vector<vector<double>> cation_pos_y(nstep, vector<double>(ncation));
    vector<vector<double>> cation_pos_z(nstep, vector<double>(ncation));
    {
        vector<double> ux(ncation), uy(ncation), uz(ncation);
        for (int j = 0; j < ncation; j++)
        {
            ux[j] = cation_com_x_t[0][j]; uy[j] = cation_com_y_t[0][j]; uz[j] = cation_com_z_t[0][j];
            cation_pos_x[0][j] = ux[j] - sys_unwrapped_x[0];
            cation_pos_y[0][j] = uy[j] - sys_unwrapped_y[0];
            cation_pos_z[0][j] = uz[j] - sys_unwrapped_z[0];
        }
        for (int i = 1; i < nstep; i++)
        {
            for (int j = 0; j < ncation; j++)
            {
                double dx = cation_com_x_t[i][j] - cation_com_x_t[i-1][j];
                double dy = cation_com_y_t[i][j] - cation_com_y_t[i-1][j];
                double dz = cation_com_z_t[i][j] - cation_com_z_t[i-1][j];
                if (dx > hL) dx -= L; else if (dx <= -hL) dx += L;
                if (dy > hL) dy -= L; else if (dy <= -hL) dy += L;
                if (dz > hL) dz -= L; else if (dz <= -hL) dz += L;
                ux[j] += dx; uy[j] += dy; uz[j] += dz;
                cation_pos_x[i][j] = ux[j] - sys_unwrapped_x[i];
                cation_pos_y[i][j] = uy[j] - sys_unwrapped_y[i];
                cation_pos_z[i][j] = uz[j] - sys_unwrapped_z[i];
            }
        }
    }

    // ===== TIME-UNWRAP ANION1 COM =====
    vector<vector<double>> anion1_pos_x(nstep, vector<double>(nanion1));
    vector<vector<double>> anion1_pos_y(nstep, vector<double>(nanion1));
    vector<vector<double>> anion1_pos_z(nstep, vector<double>(nanion1));
    {
        vector<double> ux(nanion1), uy(nanion1), uz(nanion1);
        for (int j = 0; j < nanion1; j++)
        {
            ux[j] = anion1_com_x_t[0][j]; uy[j] = anion1_com_y_t[0][j]; uz[j] = anion1_com_z_t[0][j];
            anion1_pos_x[0][j] = ux[j] - sys_unwrapped_x[0];
            anion1_pos_y[0][j] = uy[j] - sys_unwrapped_y[0];
            anion1_pos_z[0][j] = uz[j] - sys_unwrapped_z[0];
        }
        for (int i = 1; i < nstep; i++)
        {
            for (int j = 0; j < nanion1; j++)
            {
                double dx = anion1_com_x_t[i][j] - anion1_com_x_t[i-1][j];
                double dy = anion1_com_y_t[i][j] - anion1_com_y_t[i-1][j];
                double dz = anion1_com_z_t[i][j] - anion1_com_z_t[i-1][j];
                if (dx > hL) dx -= L; else if (dx <= -hL) dx += L;
                if (dy > hL) dy -= L; else if (dy <= -hL) dy += L;
                if (dz > hL) dz -= L; else if (dz <= -hL) dz += L;
                ux[j] += dx; uy[j] += dy; uz[j] += dz;
                anion1_pos_x[i][j] = ux[j] - sys_unwrapped_x[i];
                anion1_pos_y[i][j] = uy[j] - sys_unwrapped_y[i];
                anion1_pos_z[i][j] = uz[j] - sys_unwrapped_z[i];
            }
        }
    }

    // ===== TIME-UNWRAP ANION2 COM ======
    vector<vector<double>> anion2_pos_x(nstep, vector<double>(nanion2));
    vector<vector<double>> anion2_pos_y(nstep, vector<double>(nanion2));
    vector<vector<double>> anion2_pos_z(nstep, vector<double>(nanion2));
    {
        vector<double> ux(nanion2), uy(nanion2), uz(nanion2);
        for (int j = 0; j < nanion2; j++)
        {
            ux[j] = anion2_com_x_t[0][j]; uy[j] = anion2_com_y_t[0][j]; uz[j] = anion2_com_z_t[0][j];
            anion2_pos_x[0][j] = ux[j] - sys_unwrapped_x[0];
            anion2_pos_y[0][j] = uy[j] - sys_unwrapped_y[0];
            anion2_pos_z[0][j] = uz[j] - sys_unwrapped_z[0];
        }
        for (int i = 1; i < nstep; i++)
        {
            for (int j = 0; j < nanion2; j++)
            {
                double dx = anion2_com_x_t[i][j] - anion2_com_x_t[i-1][j];
                double dy = anion2_com_y_t[i][j] - anion2_com_y_t[i-1][j];
                double dz = anion2_com_z_t[i][j] - anion2_com_z_t[i-1][j];
                if (dx > hL) dx -= L; else if (dx <= -hL) dx += L;
                if (dy > hL) dy -= L; else if (dy <= -hL) dy += L;
                if (dz > hL) dz -= L; else if (dz <= -hL) dz += L;
                ux[j] += dx; uy[j] += dy; uz[j] += dz;
                anion2_pos_x[i][j] = ux[j] - sys_unwrapped_x[i];
                anion2_pos_y[i][j] = uy[j] - sys_unwrapped_y[i];
                anion2_pos_z[i][j] = uz[j] - sys_unwrapped_z[i];
            }
        }
    }

    cout << "Computing block-averaged MSD/correlation curves (9 quantities)..." << endl;
    
    // ===================================================================
    // ===== BLOCK-AVERAGED MSD AND DISTINCT-CORRELATION CALCULATION =====
    // ===================================================================
    
    int nblocks = 10;
    int block_size = nstep / nblocks;
    if (nstep % nblocks != 0)
        cout << "Warning: nstep (" << nstep << ") is not evenly divisible by nblocks ("
             << nblocks << "); the last " << (nstep % nblocks)
             << " frame(s) will be dropped from the last block." << endl;

    int STRIDE_DT = 1;
    int STRIDE_ORIGIN = 1;

    vector<double> sum_self_cation(block_size, 0.0);    // D+
    vector<double> sum_self_anion1(block_size, 0.0);    // D1
    vector<double> sum_self_anion2(block_size, 0.0);    // D2
    vector<double> sum_cc(block_size, 0.0);             // D++
    vector<double> sum_11(block_size, 0.0);             // D11
    vector<double> sum_22(block_size, 0.0);             // D22
    vector<double> sum_c1(block_size, 0.0);             // D+1
    vector<double> sum_c2(block_size, 0.0);             // D+2
    vector<double> sum_12(block_size, 0.0);             // D12
    vector<double> time_arr(block_size, 0.0);
    vector<vector<vector<double>>> all_msd(9, vector<vector<double>>(nblocks, vector<double>(block_size, 0.0)));

    //================================================
    //================= THREADS ======================
    //================================================
    
    omp_set_num_threads(20); 

    for (int blk = 0; blk < nblocks; blk++)
    {
        int block_start = blk * block_size;
        cout << "  Block " << blk << " : frames [" << block_start << ", "
             << (block_start + block_size) << ")" << endl;

        vector<double> blk_self_cation(block_size, 0.0);
        vector<double> blk_self_anion1(block_size, 0.0);
        vector<double> blk_self_anion2(block_size, 0.0);
        vector<double> blk_cc(block_size, 0.0);
        vector<double> blk_11(block_size, 0.0);
        vector<double> blk_22(block_size, 0.0);
        vector<double> blk_c1(block_size, 0.0);
        vector<double> blk_c2(block_size, 0.0);
        vector<double> blk_12(block_size, 0.0);

        #pragma omp parallel for schedule(dynamic)
        for (int dt = 0; dt < block_size; dt += STRIDE_DT)
        {
            long long numorigins = 0;
            double sumSelfCation = 0.0, sumSelfAnion1 = 0.0, sumSelfAnion2 = 0.0;
            double sumCC = 0.0, sum11 = 0.0, sum22 = 0.0;
            double sumC1 = 0.0, sumC2 = 0.0, sum12 = 0.0;

            for (int ii = 0; ii < block_size - dt; ii += STRIDE_ORIGIN)
            {
                int i = block_start + ii;
                int k = i + dt;
                numorigins++;

                // ---- collective cation displacement, self sum ----
                double Rcx = 0.0, Rcy = 0.0, Rcz = 0.0, selfC = 0.0;
                for (int j = 0; j < ncation; j++)
                {
                    double dx = cation_pos_x[k][j] - cation_pos_x[i][j];
                    double dy = cation_pos_y[k][j] - cation_pos_y[i][j];
                    double dz = cation_pos_z[k][j] - cation_pos_z[i][j];
                    Rcx += dx; Rcy += dy; Rcz += dz;
                    selfC += dx*dx + dy*dy + dz*dz;
                }

                // ---- collective anion1 displacement, self sum ----
                double R1x = 0.0, R1y = 0.0, R1z = 0.0, self1 = 0.0;
                for (int j = 0; j < nanion1; j++)
                {
                    double dx = anion1_pos_x[k][j] - anion1_pos_x[i][j];
                    double dy = anion1_pos_y[k][j] - anion1_pos_y[i][j];
                    double dz = anion1_pos_z[k][j] - anion1_pos_z[i][j];
                    R1x += dx; R1y += dy; R1z += dz;
                    self1 += dx*dx + dy*dy + dz*dz;
                }

                // ---- collective anion2 displacement, self sum ----
                double R2x = 0.0, R2y = 0.0, R2z = 0.0, self2 = 0.0;
                for (int j = 0; j < nanion2; j++)
                {
                    double dx = anion2_pos_x[k][j] - anion2_pos_x[i][j];
                    double dy = anion2_pos_y[k][j] - anion2_pos_y[i][j];
                    double dz = anion2_pos_z[k][j] - anion2_pos_z[i][j];
                    R2x += dx; R2y += dy; R2z += dz;
                    self2 += dx*dx + dy*dy + dz*dz;
                }

                sumSelfCation += selfC;
                sumSelfAnion1 += self1;
                sumSelfAnion2 += self2;

                double RC2 = Rcx*Rcx + Rcy*Rcy + Rcz*Rcz;
                double R1sq = R1x*R1x + R1y*R1y + R1z*R1z;
                double R2sq = R2x*R2x + R2y*R2y + R2z*R2z;

                sumCC += (RC2  - selfC);   // D++  same-species distinct
                sum11 += (R1sq - self1);   // D11  same-species distinct
                sum22 += (R2sq - self2);   // D22  same-species distinct

                sumC1 += Rcx*R1x + Rcy*R1y + Rcz*R1z; // D+1 cross
                sumC2 += Rcx*R2x + Rcy*R2y + Rcz*R2z; // D+2 cross
                sum12 += R1x*R2x + R1y*R2y + R1z*R2z; // D12 cross
            }

            blk_self_cation[dt] = sumSelfCation / (double(numorigins) * double(ncation));
            blk_self_anion1[dt] = sumSelfAnion1 / (double(numorigins) * double(nanion1));
            blk_self_anion2[dt] = sumSelfAnion2 / (double(numorigins) * double(nanion2));

            blk_cc[dt] = sumCC / (double(numorigins) * double(ncation));
            blk_11[dt] = sum11 / (double(numorigins) * double(nanion1));
            blk_22[dt] = sum22 / (double(numorigins) * double(nanion2));

            blk_c1[dt] = sumC1 / (double(numorigins) * sqrt(double(ncation) * double(nanion1)));
            blk_c2[dt] = sumC2 / (double(numorigins) * sqrt(double(ncation) * double(nanion2)));
            blk_12[dt] = sum12 / (double(numorigins) * sqrt(double(nanion1) * double(nanion2)));
        }

        for (int dt = 0; dt < block_size; dt += STRIDE_DT)
        {
            all_msd[0][blk][dt] = blk_self_cation[dt];
            all_msd[1][blk][dt] = blk_self_anion1[dt];
            all_msd[2][blk][dt] = blk_self_anion2[dt];
            all_msd[3][blk][dt] = blk_cc[dt];
            all_msd[4][blk][dt] = blk_11[dt];
            all_msd[5][blk][dt] = blk_22[dt];
            all_msd[6][blk][dt] = blk_c1[dt];
            all_msd[7][blk][dt] = blk_c2[dt];
            all_msd[8][blk][dt] = blk_12[dt];
            
            sum_self_cation[dt] += blk_self_cation[dt];
            sum_self_anion1[dt] += blk_self_anion1[dt];
            sum_self_anion2[dt] += blk_self_anion2[dt];
            sum_cc[dt] += blk_cc[dt];
            sum_11[dt] += blk_11[dt];
            sum_22[dt] += blk_22[dt];
            sum_c1[dt] += blk_c1[dt];
            sum_c2[dt] += blk_c2[dt];
            sum_12[dt] += blk_12[dt];
            time_arr[dt] = dt * timestep;
        }
    }

    // ===== AVERAGE ACROSS BLOCKS =====
    vector<double> avg_self_cation(block_size), avg_self_anion1(block_size), avg_self_anion2(block_size);
    vector<double> avg_cc(block_size), avg_11(block_size), avg_22(block_size);
    vector<double> avg_c1(block_size), avg_c2(block_size), avg_12(block_size);

    for (int dt = 0; dt < block_size; dt += STRIDE_DT)
    {
        avg_self_cation[dt] = sum_self_cation[dt] / nblocks;
        avg_self_anion1[dt] = sum_self_anion1[dt] / nblocks;
        avg_self_anion2[dt] = sum_self_anion2[dt] / nblocks;
        avg_cc[dt] = sum_cc[dt] / nblocks;
        avg_11[dt] = sum_11[dt] / nblocks;
        avg_22[dt] = sum_22[dt] / nblocks;
        avg_c1[dt] = sum_c1[dt] / nblocks;
        avg_c2[dt] = sum_c2[dt] / nblocks;
        avg_12[dt] = sum_12[dt] / nblocks;
    }

    // ========================================================================
    // DIRECT EINSTEIN-HELFAND CALCULATION (ADDITIONAL OUTPUT)
    // ========================================================================

    cout << "Computing DIRECT Einstein-Helfand charge MSD..." << endl;

    const double z_cation_EH = +0.8;
    const double z_anion1_EH = -0.8;
    const double z_anion2_EH = -0.8;

    vector<double> sum_EH(block_size, 0.0);
    vector<double> time_EH(block_size, 0.0);
    vector<vector<double>> all_blk_EH(nblocks, vector<double>(block_size, 0.0));

    for (int blk = 0; blk < nblocks; blk++)
    {
        int block_start = blk * block_size;

        vector<double> blk_EH(block_size, 0.0);

        #pragma omp parallel for schedule(dynamic)
        for (int dt = 0; dt < block_size; dt += STRIDE_DT)
        {
            long long numorigins = 0;
            double sumEH = 0.0;

            for (int ii = 0; ii < block_size - dt; ii += STRIDE_ORIGIN)
            {
                int i = block_start + ii;
                int k = i + dt;
                numorigins++;

                double Rcx = 0.0, Rcy = 0.0, Rcz = 0.0;
                for (int j = 0; j < ncation; j++)
                {
                    Rcx += cation_pos_x[k][j] - cation_pos_x[i][j];
                    Rcy += cation_pos_y[k][j] - cation_pos_y[i][j];
                    Rcz += cation_pos_z[k][j] - cation_pos_z[i][j];
                }

                double R1x = 0.0, R1y = 0.0, R1z = 0.0;
                for (int j = 0; j < nanion1; j++)
                {
                    R1x += anion1_pos_x[k][j] - anion1_pos_x[i][j];
                    R1y += anion1_pos_y[k][j] - anion1_pos_y[i][j];
                    R1z += anion1_pos_z[k][j] - anion1_pos_z[i][j];
                }

                double R2x = 0.0, R2y = 0.0, R2z = 0.0;
                for (int j = 0; j < nanion2; j++)
                {
                    R2x += anion2_pos_x[k][j] - anion2_pos_x[i][j];
                    R2y += anion2_pos_y[k][j] - anion2_pos_y[i][j];
                    R2z += anion2_pos_z[k][j] - anion2_pos_z[i][j];
                }

                double Mx = z_cation_EH * Rcx
                          + z_anion1_EH * R1x
                          + z_anion2_EH * R2x;

                double My = z_cation_EH * Rcy
                          + z_anion1_EH * R1y
                          + z_anion2_EH * R2y;

                double Mz = z_cation_EH * Rcz
                          + z_anion1_EH * R1z
                          + z_anion2_EH * R2z;

                sumEH += Mx * Mx + My * My + Mz * Mz;
            }

            if (numorigins > 0)
                blk_EH[dt] = sumEH / double(numorigins);
        }

        for (int dt = 0; dt < block_size; dt += STRIDE_DT)
        {
            all_blk_EH[blk][dt] = blk_EH[dt];
            sum_EH[dt] += blk_EH[dt];
            time_EH[dt] = dt * timestep;
        }
    }

    vector<double> avg_EH(block_size, 0.0);

    for (int dt = 0; dt < block_size; dt += STRIDE_DT)
        avg_EH[dt] = sum_EH[dt] / double(nblocks);

    auto write_with_blocks = [&](const string& filename,
                                 const string& extra_header,
                                 const vector<double>& avg_curve,
                                 const vector<vector<double>>& blk_curves)
    {
        ofstream out(filename);
        out << extra_header;
        out << "# time(ps)\tavg(pm^2)\tsem(pm^2)";
        for (int b = 0; b < nblocks; b++) out << "\tblk" << b;
        out << "\n" << setprecision(12);

        for (int dt = 0; dt < block_size; dt += STRIDE_DT)
        {
            double mean = avg_curve[dt] * 10000.0;
            double sq_sum = 0.0;
            for (int b = 0; b < nblocks; b++)
            {
                double val = blk_curves[b][dt] * 10000.0;
                sq_sum += (val - mean) * (val - mean);
            }
            double sem = (nblocks > 1) ? sqrt(sq_sum / (double(nblocks) * (nblocks - 1))) : 0.0;

            out << time_arr[dt] << "\t" << mean << "\t" << sem;
            for (int b = 0; b < nblocks; b++)
                out << "\t" << (blk_curves[b][dt] * 10000.0);
            out << "\n";
        }
        out.close();
    };

    ostringstream eh_hdr;
    eh_hdr << "# Direct Einstein-Helfand charge MSD\n"
           << "# L(Angstrom)=" << setprecision(10) << L
           << "\tV(Angstrom^3)=" << (L * L * L) << "\n";
    write_with_blocks("Eh_direct_blockavg.dat", eh_hdr.str(), avg_EH, all_blk_EH);

    cout << "Saved direct EH file:" << endl;
    cout << "  Eh_direct_blockavg.dat" << endl << endl;

    const string msd_files[9] = {
        "msd_cation_self_blockavg.dat", "msd_anion1_self_blockavg.dat", "msd_anion2_self_blockavg.dat",
        "msd_cat_cat_blockavg.dat",     "msd_an1_an1_blockavg.dat",     "msd_an2_an2_blockavg.dat",
        "msd_cat_an1_blockavg.dat",     "msd_cat_an2_blockavg.dat",     "msd_an1_an2_blockavg.dat" };
    const vector<double>* avg_ptrs[9] = {
        &avg_self_cation, &avg_self_anion1, &avg_self_anion2,
        &avg_cc,          &avg_11,          &avg_22,
        &avg_c1,          &avg_c2,          &avg_12 };
    string msd_header = "# Averaged over " + to_string(nblocks) + " blocks.\n";

    for (int q = 0; q < 9; q++)
        write_with_blocks(msd_files[q], msd_header, *avg_ptrs[q], all_msd[q]);

    cout << "Saved 9 block-averaged files (with SEM and individual block columns):" << endl;
    cout << "  msd_cation_self_blockavg.dat  (D+)"  << endl;
    cout << "  msd_anion1_self_blockavg.dat  (D1)"  << endl;
    cout << "  msd_anion2_self_blockavg.dat  (D2)"  << endl;
    cout << "  msd_cat_cat_blockavg.dat      (D++)" << endl;
    cout << "  msd_an1_an1_blockavg.dat      (D11)" << endl;
    cout << "  msd_an2_an2_blockavg.dat      (D22)" << endl;
    cout << "  msd_cat_an1_blockavg.dat      (D+1)" << endl;
    cout << "  msd_cat_an2_blockavg.dat      (D+2)" << endl;
    cout << "  msd_an1_an2_blockavg.dat      (D12)" << endl;
    return 0;
}

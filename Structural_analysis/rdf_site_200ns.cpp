//
// Description:
// Computes site-site Radial Distribution Functions g(r) and cumulative Coordination
// Numbers (CN) from a multi-component MD trajectory ("input.xyz") using OpenMP
// parallelization and periodic boundary conditions.
// 
// System Composition:
//   Species G (1000 molecules, 10 atoms/mol): separates N sites and H sites
//   Species A4 (350 molecules, 5 atoms/mol, e.g., AlCl4-): separates Al and Cl sites
//   Species A7 (650 molecules, 9 atoms/mol, e.g., Al2Cl7-): separates Al, terminal Cl, and bridging Cl sites
// 
// Output:
//   RDF files: rdf_HCl4.txt, rdf_HCl7_term.txt, rdf_HCl7_brid.txt, rdf_NAl4.txt, rdf_NAl7.txt
//   Coordination number files: cn_HCl4.txt, cn_HCl7_term.txt, cn_HCl7_brid.txt, cn_NAl4.txt, cn_NAl7.txt
//
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using std::array;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;

namespace {

struct FrameData {
    vector<double> g_hx, g_hy, g_hz;
    vector<double> g_nx, g_ny, g_nz;

    vector<double> a4_alx, a4_aly, a4_alz;
    vector<double> a4_clx, a4_cly, a4_clz;

    vector<double> a7_alx, a7_aly, a7_alz;
    vector<double> a7_termx, a7_termy, a7_termz;
    vector<double> a7_bridx, a7_bridy, a7_bridz;
};

inline double pbc_delta(double d, double hl, double box_len) {
    if (d > hl) return d - box_len;
    if (d < -hl) return d + box_len;
    return d;
}

inline void add_pair(
    double x1, double y1, double z1,
    double x2, double y2, double z2,
    double hl, double box_len, double inv_dr, int nbin,
    vector<double>& hist
) {
    double dx = pbc_delta(x1 - x2, hl, box_len);
    double dy = pbc_delta(y1 - y2, hl, box_len);
    double dz = pbc_delta(z1 - z2, hl, box_len);

    const double r2 = dx * dx + dy * dy + dz * dz;
    const int bin = static_cast<int>(std::sqrt(r2) * inv_dr);
    if (bin < nbin) {
        hist[bin] += 1.0;
    }
}

bool read_frame(
    ifstream& fin,
    FrameData& frame,
    int nG, int naG,
    int nA4, int naA4,
    int nA7, int naA7
) {
    int natom = 0;
    if (!(fin >> natom)) {
        return false;
    }

    string line;
    std::getline(fin, line);
    std::getline(fin, line);

    string atom;
    double x = 0.0, y = 0.0, z = 0.0;

    frame.g_hx.clear(); frame.g_hy.clear(); frame.g_hz.clear();
    frame.g_nx.clear(); frame.g_ny.clear(); frame.g_nz.clear();
    frame.a4_alx.clear(); frame.a4_aly.clear(); frame.a4_alz.clear();
    frame.a4_clx.clear(); frame.a4_cly.clear(); frame.a4_clz.clear();
    frame.a7_alx.clear(); frame.a7_aly.clear(); frame.a7_alz.clear();
    frame.a7_termx.clear(); frame.a7_termy.clear(); frame.a7_termz.clear();
    frame.a7_bridx.clear(); frame.a7_bridy.clear(); frame.a7_bridz.clear();

    frame.g_hx.reserve(nG * 6);
    frame.g_hy.reserve(nG * 6);
    frame.g_hz.reserve(nG * 6);
    frame.g_nx.reserve(nG * 3);
    frame.g_ny.reserve(nG * 3);
    frame.g_nz.reserve(nG * 3);

    frame.a4_alx.reserve(nA4);
    frame.a4_aly.reserve(nA4);
    frame.a4_alz.reserve(nA4);
    frame.a4_clx.reserve(nA4 * 4);
    frame.a4_cly.reserve(nA4 * 4);
    frame.a4_clz.reserve(nA4 * 4);

    frame.a7_alx.reserve(nA7 * 2);
    frame.a7_aly.reserve(nA7 * 2);
    frame.a7_alz.reserve(nA7 * 2);
    frame.a7_termx.reserve(nA7 * 6);
    frame.a7_termy.reserve(nA7 * 6);
    frame.a7_termz.reserve(nA7 * 6);
    frame.a7_bridx.reserve(nA7);
    frame.a7_bridy.reserve(nA7);
    frame.a7_bridz.reserve(nA7);

    for (int i = 0; i < nG; ++i) {
        for (int j = 0; j < naG; ++j) {
            fin >> atom >> x >> y >> z;
            if (j >= 4) {
                frame.g_hx.push_back(x);
                frame.g_hy.push_back(y);
                frame.g_hz.push_back(z);
            } else if (j >= 1) {
                frame.g_nx.push_back(x);
                frame.g_ny.push_back(y);
                frame.g_nz.push_back(z);
            }
        }
    }

    for (int i = 0; i < nA4; ++i) {
        for (int j = 0; j < naA4; ++j) {
            fin >> atom >> x >> y >> z;
            if (j == 0) {
                frame.a4_alx.push_back(x);
                frame.a4_aly.push_back(y);
                frame.a4_alz.push_back(z);
            } else {
                frame.a4_clx.push_back(x);
                frame.a4_cly.push_back(y);
                frame.a4_clz.push_back(z);
            }
        }
    }

    for (int i = 0; i < nA7; ++i) {
        for (int j = 0; j < naA7; ++j) {
            fin >> atom >> x >> y >> z;
            if (j == 0 || j == 3) {
                frame.a7_alx.push_back(x);
                frame.a7_aly.push_back(y);
                frame.a7_alz.push_back(z);
            } else if (j == 4) {
                frame.a7_bridx.push_back(x);
                frame.a7_bridy.push_back(y);
                frame.a7_bridz.push_back(z);
            } else {
                frame.a7_termx.push_back(x);
                frame.a7_termy.push_back(y);
                frame.a7_termz.push_back(z);
            }
        }
    }

    return true;
}

void process_frame(
    const FrameData& frame,
    double hl, double box_len, double inv_dr, int nbin,
    vector<double>& g_HCl4,
    vector<double>& g_HCl7_term,
    vector<double>& g_HCl7_brid,
    vector<double>& g_NAl4,
    vector<double>& g_NAl7
) {
    const int n_h = static_cast<int>(frame.g_hx.size());
    const int n_n = static_cast<int>(frame.g_nx.size());
    const int n_a4_al = static_cast<int>(frame.a4_alx.size());
    const int n_a4_cl = static_cast<int>(frame.a4_clx.size());
    const int n_a7_al = static_cast<int>(frame.a7_alx.size());
    const int n_a7_term = static_cast<int>(frame.a7_termx.size());
    const int n_a7_brid = static_cast<int>(frame.a7_bridx.size());

    for (int h = 0; h < n_h; ++h) {
        const double hx = frame.g_hx[h];
        const double hy = frame.g_hy[h];
        const double hz = frame.g_hz[h];

        for (int j = 0; j < n_a4_cl; ++j) {
            add_pair(hx, hy, hz,
                     frame.a4_clx[j], frame.a4_cly[j], frame.a4_clz[j],
                     hl, box_len, inv_dr, nbin, g_HCl4);
        }

        for (int j = 0; j < n_a7_term; ++j) {
            add_pair(hx, hy, hz,
                     frame.a7_termx[j], frame.a7_termy[j], frame.a7_termz[j],
                     hl, box_len, inv_dr, nbin, g_HCl7_term);
        }

        for (int j = 0; j < n_a7_brid; ++j) {
            add_pair(hx, hy, hz,
                     frame.a7_bridx[j], frame.a7_bridy[j], frame.a7_bridz[j],
                     hl, box_len, inv_dr, nbin, g_HCl7_brid);
        }
    }

    for (int n = 0; n < n_n; ++n) {
        const double nx = frame.g_nx[n];
        const double ny = frame.g_ny[n];
        const double nz = frame.g_nz[n];

        for (int j = 0; j < n_a4_al; ++j) {
            add_pair(nx, ny, nz,
                     frame.a4_alx[j], frame.a4_aly[j], frame.a4_alz[j],
                     hl, box_len, inv_dr, nbin, g_NAl4);
        }

        for (int j = 0; j < n_a7_al; ++j) {
            add_pair(nx, ny, nz,
                     frame.a7_alx[j], frame.a7_aly[j], frame.a7_alz[j],
                     hl, box_len, inv_dr, nbin, g_NAl7);
        }
    }
}

}  // namespace

int main() {
    const int nf = 10000;

    const int nG = 1000, naG = 10;
    const int nA4 = 350, naA4 = 5;
    const int nA7 = 650, naA7 = 9;

    const double dr = 0.02;
    const double box_len = 73.7152;
    const double hl = box_len / 2.0;
    const double volume = box_len * box_len * box_len;
    const double inv_dr = 1.0 / dr;
    const int nbin = static_cast<int>(hl / dr);

    vector<double> g_HCl4(nbin, 0.0), g_HCl7_term(nbin, 0.0), g_HCl7_brid(nbin, 0.0);
    vector<double> g_NAl4(nbin, 0.0), g_NAl7(nbin, 0.0);

    ifstream fin("input.xyz");
    if (!fin) {
        std::cerr << "Failed to open input.xyz\n";
        return 1;
    }

    const int chunk_size = 64;
    vector<FrameData> frames(chunk_size);
    int frames_done = 0;

    while (frames_done < nf) {
        int loaded = 0;
        while (loaded < chunk_size && frames_done + loaded < nf && read_frame(fin, frames[loaded], nG, naG, nA4, naA4, nA7, naA7)) {
            ++loaded;
        }
        if (loaded == 0) {
            break;
        }

#pragma omp parallel
        {
            vector<double> local_HCl4(nbin, 0.0), local_HCl7_term(nbin, 0.0), local_HCl7_brid(nbin, 0.0);
            vector<double> local_NAl4(nbin, 0.0), local_NAl7(nbin, 0.0);

#pragma omp for schedule(static)
            for (int idx = 0; idx < loaded; ++idx) {
                process_frame(frames[idx], hl, box_len, inv_dr, nbin,
                              local_HCl4, local_HCl7_term, local_HCl7_brid,
                              local_NAl4, local_NAl7);
            }

#pragma omp critical
            {
                for (int b = 0; b < nbin; ++b) {
                    g_HCl4[b] += local_HCl4[b];
                    g_HCl7_term[b] += local_HCl7_term[b];
                    g_HCl7_brid[b] += local_HCl7_brid[b];
                    g_NAl4[b] += local_NAl4[b];
                    g_NAl7[b] += local_NAl7[b];
                }
            }
        }

        frames_done += loaded;
    }

    const double rho_Cl4 = (nA4 * 4.0) / volume;
    const double rho_Cl7_term = (nA7 * 6.0) / volume;
    const double rho_Cl7_brid = nA7 / volume;
    const double rho_Al4 = nA4 / volume;
    const double rho_Al7 = (nA7 * 2.0) / volume;

    ofstream f1("rdf_HCl4.txt"), f2("rdf_HCl7_term.txt"), f3("rdf_HCl7_brid.txt");
    ofstream f4("rdf_NAl4.txt"), f5("rdf_NAl7.txt");

    ofstream c1("cn_HCl4.txt"), c2("cn_HCl7_term.txt"), c3("cn_HCl7_brid.txt");
    ofstream c4("cn_NAl4.txt"), c5("cn_NAl7.txt");

    double cum1 = 0.0, cum2 = 0.0, cum3 = 0.0, cum4 = 0.0, cum5 = 0.0;

    for (int b = 1; b < nbin; ++b) {
        const double r = b * dr;
        const double shell = 4.0 * M_PI * r * r * dr;

        const double g1 = g_HCl4[b] / (frames_done * nG * 6.0 * rho_Cl4 * shell);
        const double g2 = g_HCl7_term[b] / (frames_done * nG * 6.0 * rho_Cl7_term * shell);
        const double g3 = g_HCl7_brid[b] / (frames_done * nG * 6.0 * rho_Cl7_brid * shell);
        const double g4 = g_NAl4[b] / (frames_done * nG * 3.0 * rho_Al4 * shell);
        const double g5 = g_NAl7[b] / (frames_done * nG * 3.0 * rho_Al7 * shell);

        f1 << r << '\t' << g1 << '\n';
        f2 << r << '\t' << g2 << '\n';
        f3 << r << '\t' << g3 << '\n';
        f4 << r << '\t' << g4 << '\n';
        f5 << r << '\t' << g5 << '\n';

        cum1 += g1 * shell * rho_Cl4;
        cum2 += g2 * shell * rho_Cl7_term;
        cum3 += g3 * shell * rho_Cl7_brid;
        cum4 += g4 * shell * rho_Al4;
        cum5 += g5 * shell * rho_Al7;

        c1 << r << '\t' << cum1 << '\n';
        c2 << r << '\t' << cum2 << '\n';
        c3 << r << '\t' << cum3 << '\n';
        c4 << r << '\t' << cum4 << '\n';
        c5 << r << '\t' << cum5 << '\n';
    }

    return 0;
}

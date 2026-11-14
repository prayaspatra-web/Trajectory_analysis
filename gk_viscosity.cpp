// Green-Kubo shear viscosity analysis for GROMACS pressure-tensor XVG files.
//
// Expected XVG data columns:
// time(ps)  Pxx(bar)  Pyy(bar)  Pzz(bar)  Pxy(bar)  Pzx(bar)  Pyz(bar)
//
// Compile:
//   g++ -O3 -std=c++17 -Wall -Wextra -pedantic gk_viscosity_nvt.cpp -o gk_viscosity
//
// Run:
//   ./gk_viscosity run_pressure.xvg VOLUME_A3 TEMPERATURE_K [NBLOCKS] [MAX_LAG_PS]
//
#include <algorithm>
#include <cmath>
#include <complex>
#include <cctype>
#include <map>
#include <regex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr double kB = 1.380649e-23;
constexpr double bar_to_pa = 1.0e5;
constexpr double A3_to_m3 = 1.0e-30;
constexpr double ps_to_s = 1.0e-12;

struct Row {
    double time_ps;
    double pxy_bar;
    double pxz_bar;
    double pyz_bar;
};

struct Stats {
    double mean = 0.0;
    double sd = 0.0;
    double sem = 0.0;
};

Stats calculate_stats(const std::vector<double>& x) {
    Stats s;
    if (x.empty()) return s;
    s.mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    if (x.size() > 1) {
        double sumsq = 0.0;
        for (double v : x) sumsq += (v - s.mean) * (v - s.mean);
        s.sd = std::sqrt(sumsq / static_cast<double>(x.size() - 1));
        s.sem = s.sd / std::sqrt(static_cast<double>(x.size()));
    }
    return s;
}

std::size_t next_power_of_two(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

void fft(std::vector<std::complex<double>>& a, bool inverse) {
    const std::size_t n = a.size();

    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    const double pi = std::acos(-1.0);
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double angle = 2.0 * pi / static_cast<double>(len) *
                             (inverse ? 1.0 : -1.0);
        const std::complex<double> wlen(std::cos(angle), std::sin(angle));

        for (std::size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t j = 0; j < len / 2; ++j) {
                const auto u = a[i + j];
                const auto v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
        if (len > n / 2) break;
    }

    if (inverse) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (auto& v : a) v *= inv_n;
    }
}

std::vector<double> autocorrelation_fft(const std::vector<double>& x,
                                        std::size_t max_lag) {
    const std::size_t n = x.size();
    const std::size_t nfft = next_power_of_two(2 * n);

    std::vector<std::complex<double>> a(nfft, {0.0, 0.0});
    for (std::size_t i = 0; i < n; ++i) a[i] = {x[i], 0.0};

    fft(a, false);
    for (auto& v : a) v = std::complex<double>(std::norm(v), 0.0);
    fft(a, true);

    std::vector<double> acf(max_lag + 1, 0.0);
    for (std::size_t lag = 0; lag <= max_lag; ++lag) {
        acf[lag] = a[lag].real() / static_cast<double>(n - lag);
    }
    return acf;
}

std::string normalize_pressure_label(std::string s) {
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c)) out.push_back(static_cast<char>(std::toupper(c)));
    }
    if (out.rfind("PRES", 0) == 0) out.erase(0, 4);
    return out;
}

std::vector<Row> read_xvg(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open input file: " + filename);

    std::vector<Row> rows;
    std::string line;
    std::size_t line_number = 0;
    std::map<int, std::string> legends;
    const std::regex legend_re(
        R"REGEX(^\s*@\s*s([0-9]+)\s+legend\s+"([^"]+)")REGEX",
        std::regex::icase);

    while (std::getline(in, line)) {
        ++line_number;
        std::smatch match;
        if (std::regex_search(line, match, legend_re)) {
            legends[std::stoi(match[1].str())] =
                normalize_pressure_label(match[2].str());
            continue;
        }

        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;
        if (line[first] == '#' || line[first] == '@') continue;

        std::istringstream ss(line);
        std::vector<double> values;
        double value;
        while (ss >> value) values.push_back(value);

        if (values.size() < 7) {
            throw std::runtime_error(
                "Expected time plus six pressure columns at line " +
                std::to_string(line_number));
        }
        if (values.size() != 7) {
            throw std::runtime_error(
                "Expected exactly seven numeric columns at line " +
                std::to_string(line_number) + ", found " +
                std::to_string(values.size()));
        }

        if (legends.size() != 6) {
            throw std::runtime_error(
                "Could not identify all six pressure components from XVG "
                "header legends (@ sN legend).");
        }

        std::map<std::string, double> pressure;
        for (const auto& item : legends) {
            const int series_index = item.first;
            const std::string& component = item.second;
            const std::size_t value_index =
                static_cast<std::size_t>(series_index) + 1; // time is column 0

            if (series_index < 0 || value_index >= values.size()) {
                throw std::runtime_error(
                    "XVG legend series index is outside the numeric columns.");
            }
            if (component != "XX" && component != "YY" && component != "ZZ" &&
                component != "XY" && component != "YX" &&
                component != "XZ" && component != "ZX" &&
                component != "YZ" && component != "ZY") {
                throw std::runtime_error(
                    "Unrecognised pressure legend component: " + component);
            }

            std::string canonical = component;
            if (canonical == "YX") canonical = "XY";
            if (canonical == "ZX") canonical = "XZ";
            if (canonical == "ZY") canonical = "YZ";

            if (pressure.count(canonical)) {
                throw std::runtime_error(
                    "Duplicate pressure component in XVG header: " + canonical);
            }
            pressure[canonical] = values[value_index];
        }

        for (const char* required : {"XY", "XZ", "YZ"}) {
            if (!pressure.count(required)) {
                throw std::runtime_error(
                    std::string("Missing required shear component in XVG header: ") +
                    required);
            }
        }

        const double time = values[0];
        const double pxy = pressure.at("XY");
        const double pxz = pressure.at("XZ");
        const double pyz = pressure.at("YZ");
        if (!std::isfinite(time) || !std::isfinite(pxy) ||
            !std::isfinite(pxz) || !std::isfinite(pyz)) {
            throw std::runtime_error("Non-finite value at line " +
                                     std::to_string(line_number));
        }
        rows.push_back({time, pxy, pxz, pyz});
    }

    if (rows.size() < 100) {
        throw std::runtime_error("Too few data rows. Check the XVG file.");
    }
    return rows;
}

void write_header(std::ofstream& out, const std::string& header) {
    out << "# " << header << '\n';
    out << std::setprecision(12);
}

}

int main(int argc, char* argv[]) {
    try {
        if (argc < 4 || argc > 6) {
            std::cerr
                << "Usage: " << argv[0]
                << " pressure.xvg VOLUME_A3 TEMPERATURE_K [NBLOCKS] [MAX_LAG_PS]\n"
                << "Example: " << argv[0]
                << " run_pressure.xvg 418055 298.15 20 500\n";
            return 2;
        }

        const std::string input_file = argv[1];
        const double volume_A3 = std::stod(argv[2]);
        const double temperature_K = std::stod(argv[3]);
        const std::size_t nblocks =
            (argc >= 5) ? static_cast<std::size_t>(std::stoul(argv[4])) : 20;

        if (!(volume_A3 > 0.0) || !(temperature_K > 0.0) || nblocks < 2) {
            throw std::runtime_error("Volume and temperature must be positive; NBLOCKS must be >= 2.");
        }

        std::vector<Row> rows = read_xvg(input_file);
        const std::size_t npoints = rows.size();

        const double dt_ps = rows[1].time_ps - rows[0].time_ps;
        if (!(dt_ps > 0.0)) throw std::runtime_error("Time values are not increasing.");

        const double time_tol = std::max(1.0e-8, std::abs(dt_ps) * 1.0e-4);
        for (std::size_t i = 1; i < npoints; ++i) {
            const double dti = rows[i].time_ps - rows[i - 1].time_ps;
            if (dti <= 0.0) {
                throw std::runtime_error("Time column is duplicated or decreasing near row " +
                                         std::to_string(i + 1));
            }
            if (std::abs(dti - dt_ps) > time_tol) {
                throw std::runtime_error(
                    "Non-uniform time spacing detected near row " +
                    std::to_string(i + 1) +
                    ". Do not analyse this file as a uniformly sampled series.");
            }
        }

        const std::size_t block_size = npoints / nblocks;
        if (block_size < 100) throw std::runtime_error("Too few points per block.");
        const std::size_t used_points = block_size * nblocks;
        const double block_duration_ps = (block_size - 1) * dt_ps;

        std::size_t max_lag;
        if (argc >= 6) {
            const double requested_ps = std::stod(argv[5]);
            if (!(requested_ps > 0.0)) throw std::runtime_error("MAX_LAG_PS must be positive.");
            max_lag = static_cast<std::size_t>(std::llround(requested_ps / dt_ps));
        } else {
            max_lag = block_size / 2;
        }

        if (max_lag >= block_size) {
            throw std::runtime_error("Maximum lag must be shorter than a block.");
        }
        if (max_lag > block_size / 2) {
            std::cerr << "WARNING: max lag exceeds half a block. Long-lag estimates "
                         "will have relatively few time origins.\n";
        }

        std::vector<std::vector<double>> stress(3, std::vector<double>(used_points));
        for (std::size_t i = 0; i < used_points; ++i) {
            stress[0][i] = rows[i].pxy_bar;
            stress[1][i] = rows[i].pxz_bar;
            stress[2][i] = rows[i].pyz_bar;
        }

        std::ofstream diag("gk_diagnostics.txt");
        if (!diag) throw std::runtime_error("Cannot create gk_diagnostics.txt");
        diag << std::setprecision(12);
        diag << "Input file: " << input_file << '\n'
             << "Input points: " << npoints << '\n'
             << "Points used: " << used_points << " (remainder discarded: "
             << (npoints - used_points) << ")\n"
             << "Time interval (ps): " << dt_ps << '\n'
             << "Trajectory duration (ps): " << rows.back().time_ps - rows.front().time_ps << '\n'
             << "Number of blocks: " << nblocks << '\n'
             << "Points per block: " << block_size << '\n'
             << "Block duration (ps): " << block_duration_ps << '\n'
             << "Maximum lag (points): " << max_lag << '\n'
             << "Maximum lag (ps): " << max_lag * dt_ps << '\n'
             << "Volume (A^3): " << volume_A3 << '\n'
             << "Temperature (K): " << temperature_K << '\n'
             << "Pressure units assumed: bar\n"
             << "Shear columns used: Pxy, Pzx/Pxz, Pyz\n\n";

        const char* component_names[3] = {"Pxy", "Pxz_or_Pzx", "Pyz"};
        for (int c = 0; c < 3; ++c) {
            const double mean = std::accumulate(stress[c].begin(), stress[c].end(), 0.0)
                                / static_cast<double>(used_points);
            double var = 0.0;
            for (double& v : stress[c]) {
                v -= mean;
                var += v * v;
            }
            const double rms = std::sqrt(var / static_cast<double>(used_points));
            diag << component_names[c] << " global mean (bar): " << mean
                 << "\n" << component_names[c] << " fluctuation RMS (bar): " << rms << "\n";
        }

        std::vector<std::vector<std::vector<double>>> acf_component(
            3, std::vector<std::vector<double>>(nblocks,
                 std::vector<double>(max_lag + 1, 0.0)));

        std::cout << "Calculating FFT autocorrelations...\n";
        for (std::size_t b = 0; b < nblocks; ++b) {
            const std::size_t start = b * block_size;
            std::vector<double> block(block_size);

            for (int c = 0; c < 3; ++c) {
                std::copy(stress[c].begin() + static_cast<std::ptrdiff_t>(start),
                          stress[c].begin() + static_cast<std::ptrdiff_t>(start + block_size),
                          block.begin());
                acf_component[c][b] = autocorrelation_fft(block, max_lag);
            }
            std::cout << "  Completed block " << (b + 1) << "/" << nblocks << "\n";
        }

        std::vector<std::vector<double>> component_mean(3, std::vector<double>(max_lag + 1));
        std::vector<std::vector<double>> component_sd(3, std::vector<double>(max_lag + 1));
        std::vector<std::vector<double>> component_sem(3, std::vector<double>(max_lag + 1));

        for (int c = 0; c < 3; ++c) {
            for (std::size_t lag = 0; lag <= max_lag; ++lag) {
                std::vector<double> vals(nblocks);
                for (std::size_t b = 0; b < nblocks; ++b)
                    vals[b] = acf_component[c][b][lag];
                const Stats s = calculate_stats(vals);
                component_mean[c][lag] = s.mean;
                component_sd[c][lag] = s.sd;
                component_sem[c][lag] = s.sem;
            }
        }

        std::vector<std::vector<double>> block_acf(nblocks,
                                                   std::vector<double>(max_lag + 1, 0.0));
        std::vector<std::vector<double>> block_eta(nblocks,
                                                   std::vector<double>(max_lag + 1, 0.0));
        const double V_m3 = volume_A3 * A3_to_m3;
        const double prefactor = V_m3 / (kB * temperature_K);
        const double lag_dt_s = dt_ps * ps_to_s;

        for (std::size_t b = 0; b < nblocks; ++b) {
            for (std::size_t lag = 0; lag <= max_lag; ++lag) {
                const double acf_bar2 =
                    (acf_component[0][b][lag] +
                     acf_component[1][b][lag] +
                     acf_component[2][b][lag]) / 3.0;
                block_acf[b][lag] = acf_bar2;

                const double acf_Pa2 = acf_bar2 * bar_to_pa * bar_to_pa;
                if (lag == 0) {
                    block_eta[b][lag] = 0.0;
                } else {
                    const double prev_bar2 =
                        (acf_component[0][b][lag - 1] +
                         acf_component[1][b][lag - 1] +
                         acf_component[2][b][lag - 1]) / 3.0;
                    const double prev_Pa2 = prev_bar2 * bar_to_pa * bar_to_pa;
                    block_eta[b][lag] =
                        block_eta[b][lag - 1] +
                        0.5 * (prev_Pa2 + acf_Pa2) * lag_dt_s * prefactor * 1000.0;
                }
            }
        }

        {
            std::ofstream out("gk_acf_components.csv");
            write_header(out,
                "lag_ps,Cxy_mean_bar2,Cxy_block_SD_bar2,Cxy_SEM_bar2,"
                "Cxz_mean_bar2,Cxz_block_SD_bar2,Cxz_SEM_bar2,"
                "Cyz_mean_bar2,Cyz_block_SD_bar2,Cyz_SEM_bar2,"
                "Cshear_mean_bar2,Cshear_block_SD_bar2,Cshear_SEM_bar2");

            for (std::size_t lag = 0; lag <= max_lag; ++lag) {
                std::vector<double> shear_vals(nblocks);
                for (std::size_t b = 0; b < nblocks; ++b)
                    shear_vals[b] = block_acf[b][lag];
                const Stats ss = calculate_stats(shear_vals);

                out << lag * dt_ps;
                for (int c = 0; c < 3; ++c)
                    out << ',' << component_mean[c][lag]
                        << ',' << component_sd[c][lag]
                        << ',' << component_sem[c][lag];
                out << ',' << ss.mean << ',' << ss.sd << ',' << ss.sem << '\n';
            }
        }

        {
            std::ofstream out("gk_acf_by_block.csv");
            write_header(out, "lag_ps,block1_bar2,...,blockN_bar2");
            for (std::size_t lag = 0; lag <= max_lag; ++lag) {
                out << lag * dt_ps;
                for (std::size_t b = 0; b < nblocks; ++b)
                    out << ',' << block_acf[b][lag];
                out << '\n';
            }
        }

        {
            std::ofstream out("gk_viscosity_convergence.csv");
            write_header(out,
                "lag_ps,eta_mean_mPa_s,eta_block_SD_mPa_s,eta_SEM_mPa_s,"
                "relative_SEM_percent,block1_mPa_s,...,blockN_mPa_s");

            for (std::size_t lag = 0; lag <= max_lag; ++lag) {
                std::vector<double> vals(nblocks);
                for (std::size_t b = 0; b < nblocks; ++b) vals[b] = block_eta[b][lag];
                const Stats s = calculate_stats(vals);
                const double rel_sem = (std::abs(s.mean) > 1.0e-15)
                                           ? 100.0 * s.sem / std::abs(s.mean)
                                           : std::numeric_limits<double>::quiet_NaN();

                out << lag * dt_ps << ',' << s.mean << ',' << s.sd << ','
                    << s.sem << ',' << rel_sem;
                for (double v : vals) out << ',' << v;
                out << '\n';
            }
        }

        {
            std::vector<double> endpoint(nblocks);
            for (std::size_t b = 0; b < nblocks; ++b)
                endpoint[b] = block_eta[b][max_lag];
            const Stats s = calculate_stats(endpoint);
            diag << "\nEndpoint at max lag (diagnostic only; not automatically final):\n"
                 << "Lag (ps): " << max_lag * dt_ps << '\n'
                 << "Mean eta (mPa.s): " << s.mean << '\n'
                 << "Block SD (mPa.s): " << s.sd << '\n'
                 << "SEM (mPa.s): " << s.sem << '\n'
                 << "Relative SEM (%): "
                 << (std::abs(s.mean) > 1.0e-15 ? 100.0 * s.sem / std::abs(s.mean) : 0.0)
                 << '\n';
        }

        {
            std::vector<double> eta_mean(max_lag + 1, 0.0);
            for (std::size_t lag = 0; lag <= max_lag; ++lag) {
                for (std::size_t b = 0; b < nblocks; ++b)
                    eta_mean[lag] += block_eta[b][lag];
                eta_mean[lag] /= static_cast<double>(nblocks);
            }

            const std::size_t w0 = std::max<std::size_t>(1, (4 * max_lag) / 5);
            const std::size_t w1 = max_lag;
            double mean_eta_window = 0.0;
            for (std::size_t i = w0; i <= w1; ++i) mean_eta_window += eta_mean[i];
            mean_eta_window /= static_cast<double>(w1 - w0 + 1);

            const double drift = eta_mean[w1] - eta_mean[w0];
            const double drift_pct = (std::abs(mean_eta_window) > 1.0e-15)
                                         ? 100.0 * drift / std::abs(mean_eta_window)
                                         : std::numeric_limits<double>::quiet_NaN();

            diag << "\nLate-window drift diagnostic (last 20% of selected lag range):\n"
                 << "Start lag (ps): " << w0 * dt_ps << '\n'
                 << "End lag (ps): " << w1 * dt_ps << '\n'
                 << "Change in mean cumulative eta (mPa.s): " << drift << '\n'
                 << "Change relative to late-window mean (%): " << drift_pct << '\n'
                 << "Interpretation: inspect the convergence plot; this diagnostic alone "
                    "does not establish a plateau.\n";
        }

        diag.close();

        std::cout << "\nAnalysis completed.\n"
                  << "Time spacing: " << dt_ps << " ps\n"
                  << "Total input duration: " << rows.back().time_ps - rows.front().time_ps
                  << " ps\n"
                  << "Blocks: " << nblocks << " ("
                  << block_duration_ps / 1000.0 << " ns each)\n"
                  << "Maximum lag analysed: " << max_lag * dt_ps << " ps\n"
                  << "Files written:\n"
                  << "  gk_diagnostics.txt\n"
                  << "  gk_acf_components.csv\n"
                  << "  gk_acf_by_block.csv\n"
                  << "  gk_viscosity_convergence.csv\n"
                  << "\nPlot gk_viscosity_convergence.csv: eta_mean_mPa_s vs lag_ps,\n"
                  << "with eta_SEM_mPa_s or eta_block_SD_mPa_s as uncertainty.\n"
                  << "Do not report the final endpoint unless a stable plateau is visible.\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}

//
// Description:
// Calculates the continuous hydrogen-bond lifetime autocorrelation function C(t)
// for water molecules from a molecular dynamics trajectory ("traj.xyz").
// 
// Geometric H-bond criteria applied under periodic boundary conditions (PBC):
//   Donor-Acceptor (O...O) distance < 3.5 Angstroms
//   Hydrogen-Acceptor (H...O) distance < 2.5 Angstroms
//   Donor-Hydrogen-Acceptor (O-H...O) angle >= 150 degrees
// 
// Output:
//   Writes time lag and normalized H-bond survival probability to output file.
//
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <map>
#include <vector>
#include <array>
#include <unordered_map>
#include <set>
using namespace std;

namespace std {
    template<>
    struct hash<std::array<int, 3>> {
        std::size_t operator()(const std::array<int, 3>& arr) const noexcept {
            std::size_t h = 0;
            for (int i : arr) {
                h ^= std::hash<int>()(i) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            return h;
        }
    };
}

int main() {
    int nf = 1002;   
    int nw = 2166;   
    int na = 3;      
double hbond[nf],cnt[nf];
    int natom;
    string a1, a2, a3;
    string a[nw][na];
    double x[nw][na], y[nw][na], z[nw][na];
    double hb[nf][nw];
    double L = 40.266;
    double hl = L / 2.0;
    unordered_map<array<int,3>, set<int>> hbond_frame;
for (int x=0;x<nf;x++)
{
        hbond[x]=0;
        cnt[x]=0;

}

	
    ifstream my_file("traj.xyz");
    ofstream out_file("h7.txt");

    for (int n = 0; n < nf; n++) {
        my_file >> natom;
        my_file >> a1 >> a2 >> a3;

        for (int j = 0; j < nw; j++) {
            for (int k = 0; k < na; k++) {
                my_file >> a[j][k] >> x[j][k] >> y[j][k] >> z[j][k];
            }
        }

        for (int i = 0; i < nw; i++) {
            for (int j = 0; j < nw-1; j++) {
                if (i == j) continue;

                for (int k = 1; k < na; k++) { 
                    double dx = x[j][0] - x[i][0];
                    double dy = y[j][0] - y[i][0];
                    double dz = z[j][0] - z[i][0];

                    if (dx > hl) dx -= L; else if (dx < -hl) dx += L;
                    if (dy > hl) dy -= L; else if (dy < -hl) dy += L;
                    if (dz > hl) dz -= L; else if (dz < -hl) dz += L;

                    double OO_dist = sqrt(dx*dx + dy*dy + dz*dz);
                    if (OO_dist >= 3.5) continue;

                    double dhx = x[i][0] - x[i][k];
                    double dhy = y[i][0] - y[i][k];
                    double dhz = z[i][0] - z[i][k];

                    double dhx1 = x[j][0] - x[i][k];
                    double dhy1 = y[j][0] - y[i][k];
                    double dhz1 = z[j][0] - z[i][k];

                    if (dhx > hl) dhx -= L; else if (dhx < -hl) dhx += L;
                    if (dhy > hl) dhy -= L; else if (dhy < -hl) dhy += L;
                    if (dhz > hl) dhz -= L; else if (dhz < -hl) dhz += L;

                    if (dhx1 > hl) dhx1 -= L; else if (dhx1 < -hl) dhx1 += L;
                    if (dhy1 > hl) dhy1 -= L; else if (dhy1 < -hl) dhy1 += L;
                    if (dhz1 > hl) dhz1 -= L; else if (dhz1 < -hl) dhz1 += L;

                    double h_a_dist = sqrt(dhx1*dhx1 + dhy1*dhy1 + dhz1*dhz1);
                    if (h_a_dist >= 2.5) continue;
                    double dot = dhx*dhx1 + dhy*dhy1 + dhz*dhz1;
                    double mag1 = sqrt(dhx*dhx + dhy*dhy + dhz*dhz);
                    double mag2 = sqrt(dhx1*dhx1 + dhy1*dhy1 + dhz1*dhz1);

                    double costheta = dot / (mag1 * mag2);
                    costheta = max(-1.0, min(1.0, costheta));
                    double angle = acos(costheta) * (180.0 / M_PI);

		    if (angle>=150) hb[n][i]=1;
		    else hb[n][i]=0;
		}
	}
}
}
for (int t=0;t<nf;t++)
{
// cout<<"calculating for fno. "<<'\t'<<t<<endl;
 for(int dt=0;dt<nf-t;dt++)
 {
         for ( int r=0;r<nw;r++)
         {
                        if(hb[t][r]==1)
                        {
                                cnt[dt]+=1;
                                for(int v=0;v<=dt;v++)
                                {
                                        if(hb[t+v][r]==0)
                                        {
                                                break;
                                        }
                                        if(dt==v)
                                        {
                                                hbond[dt]+=hb[t][r]*hb[t+dt][r];
                                                break;
                                        }
                                        //cout<<hbond;
                                }
                        }
        }
}
}


for(int t=0;t<nf;t++)
{
        //cout<< cnt[t]<<endl;
   out_file<<2*t<<"\t"<<double (hbond[t]) /double (cnt[t])<<endl;


}
//cout<<endl;
//cout<<cnt1;



return 0;

}

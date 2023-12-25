
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_2925n1>:
	s_load_b256 s[0:7], s[0:1], null                           // 000000001700: F40C0000 F8000000
	s_mov_b32 s8, s15                                          // 000000001708: BE88000F
	s_ashr_i32 s9, s15, 31                                     // 00000000170C: 86099F0F
	v_mov_b32_e32 v1, 0                                        // 000000001710: 7E020280
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001714: 84888208
	s_waitcnt lgkmcnt(0)                                       // 000000001718: BF89FC07
	s_add_u32 s2, s2, s8                                       // 00000000171C: 80020802
	s_addc_u32 s3, s3, s9                                      // 000000001720: 82030903
	s_add_u32 s4, s4, s8                                       // 000000001724: 80040804
	s_addc_u32 s5, s5, s9                                      // 000000001728: 82050905
	s_load_b32 s10, s[2:3], null                               // 00000000172C: F4000281 F8000000
	s_load_b32 s4, s[4:5], null                                // 000000001734: F4000102 F8000000
	s_add_u32 s2, s6, s8                                       // 00000000173C: 80020806
	s_addc_u32 s3, s7, s9                                      // 000000001740: 82030907
	s_add_u32 s0, s0, s8                                       // 000000001744: 80000800
	s_load_b32 s2, s[2:3], null                                // 000000001748: F4000081 F8000000
	s_addc_u32 s1, s1, s9                                      // 000000001750: 82010901
	s_waitcnt lgkmcnt(0)                                       // 000000001754: BF89FC07
	v_add_f32_e64 v0, s10, s4                                  // 000000001758: D5030000 0000080A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001760: BF870001
	v_add_f32_e32 v0, s2, v0                                   // 000000001764: 06000002
	global_store_b32 v1, v0, s[0:1]                            // 000000001768: DC6A0000 00000001
	s_nop 0                                                    // 000000001770: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001774: BFB60003
	s_endpgm                                                   // 000000001778: BFB00000

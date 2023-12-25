
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4n107>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_mov_b32 s2, s15                                          // 000000001708: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 00000000170C: 86039F0F
	v_mov_b32_e32 v2, 0                                        // 000000001710: 7E040280
	s_lshl_b64 s[8:9], s[2:3], 1                               // 000000001714: 84888102
	s_load_b64 s[0:1], s[0:1], 0x10                            // 000000001718: F4040000 F8000010
	s_waitcnt lgkmcnt(0)                                       // 000000001720: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001724: 80060806
	s_addc_u32 s7, s7, s9                                      // 000000001728: 82070907
	global_load_u16 v0, v2, s[6:7]                             // 00000000172C: DC4A0000 00060002
	s_waitcnt vmcnt(0)                                         // 000000001734: BF8903F7
	v_readfirstlane_b32 s6, v0                                 // 000000001738: 7E0C0500
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000173C: BF870001
	s_sext_i32_i16 s8, s6                                      // 000000001740: BE880F06
	s_lshl_b64 s[6:7], s[2:3], 2                               // 000000001744: 84868202
	s_ashr_i32 s9, s8, 31                                      // 000000001748: 86099F08
	s_add_u32 s0, s0, s6                                       // 00000000174C: 80000600
	s_addc_u32 s1, s1, s7                                      // 000000001750: 82010701
	s_load_b32 s0, s[0:1], null                                // 000000001754: F4000000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000175C: BF89FC07
	s_mul_hi_u32 s1, s0, s8                                    // 000000001760: 96810800
	s_mul_i32 s6, s0, s9                                       // 000000001764: 96060900
	s_mul_i32 s0, s0, s8                                       // 000000001768: 96000800
	s_add_i32 s1, s1, s6                                       // 00000000176C: 81010601
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001770: BF8704A9
	v_dual_mov_b32 v0, s0 :: v_dual_mov_b32 v1, s1             // 000000001774: CA100000 00000001
	s_lshl_b64 s[0:1], s[2:3], 3                               // 00000000177C: 84808302
	s_add_u32 s0, s4, s0                                       // 000000001780: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001784: 82010105
	global_store_b64 v2, v[0:1], s[0:1]                        // 000000001788: DC6E0000 00000002
	s_nop 0                                                    // 000000001790: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001794: BFB60003
	s_endpgm                                                   // 000000001798: BFB00000

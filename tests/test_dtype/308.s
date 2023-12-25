
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4n146>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 000000001718: 86039F0F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 00000000171C: BF8704D9
	s_lshl_b64 s[8:9], s[2:3], 2                               // 000000001720: 84888202
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001728: 80060806
	s_addc_u32 s7, s7, s9                                      // 00000000172C: 82070907
	s_lshl_b64 s[8:9], s[2:3], 3                               // 000000001730: 84888302
	s_add_u32 s0, s0, s8                                       // 000000001734: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001738: 82010901
	s_load_b64 s[0:1], s[0:1], null                            // 00000000173C: F4040000 F8000000
	s_load_b32 s6, s[6:7], null                                // 000000001744: F4000183 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000174C: BF89FC07
	s_clz_i32_u32 s7, s1                                       // 000000001750: BE870A01
	v_cvt_f32_i32_e32 v1, s6                                   // 000000001754: 7E020A06
	s_min_u32 s7, s7, 32                                       // 000000001758: 8987A007
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000175C: BF870499
	s_lshl_b64 s[0:1], s[0:1], s7                              // 000000001760: 84800700
	s_min_u32 s0, s0, 1                                        // 000000001764: 89808100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001768: BF8704A1
	v_cvt_f16_f32_e32 v1, v1                                   // 00000000176C: 7E021501
	s_or_b32 s0, s1, s0                                        // 000000001770: 8C000001
	v_cvt_f32_u32_e32 v0, s0                                   // 000000001774: 7E000C00
	s_sub_i32 s0, 32, s7                                       // 000000001778: 818007A0
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 00000000177C: BF870481
	v_ldexp_f32 v0, v0, s0                                     // 000000001780: D71C0000 00000100
	s_lshl_b64 s[0:1], s[2:3], 1                               // 000000001788: 84808102
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000178C: BF8700A9
	s_add_u32 s0, s4, s0                                       // 000000001790: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001794: 82010105
	v_cvt_f16_f32_e32 v0, v0                                   // 000000001798: 7E001500
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000179C: BF870001
	v_add_f16_e32 v0, v1, v0                                   // 0000000017A0: 64000101 ; Error: VGPR_32_Lo128: unknown register 128
	v_mov_b32_e32 v1, 0                                        // 0000000017A4: 7E020280
	global_store_b16 v1, v0, s[0:1]                            // 0000000017A8: DC660000 00000001
	s_nop 0                                                    // 0000000017B0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017B4: BFB60003
	s_endpgm                                                   // 0000000017B8: BFB00000

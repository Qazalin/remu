
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_11_5_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s14, 7                                       // 000000001714: 9608870E
	s_mov_b32 s2, s13                                          // 000000001718: BE82000D
	s_ashr_i32 s9, s8, 31                                      // 00000000171C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001720: BF870009
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001724: 84888208
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s8, s6, s8                                       // 00000000172C: 80080806
	s_addc_u32 s7, s7, s9                                      // 000000001730: 82070907
	s_ashr_i32 s3, s13, 31                                     // 000000001734: 86039F0D
	s_mul_i32 s6, s15, 3                                       // 000000001738: 9606830F
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000173C: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001740: BF8704B9
	s_add_u32 s8, s8, s2                                       // 000000001744: 80080208
	s_addc_u32 s9, s7, s3                                      // 000000001748: 82090307
	s_ashr_i32 s7, s6, 31                                      // 00000000174C: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001750: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001754: BF870009
	s_add_u32 s0, s0, s6                                       // 000000001758: 80000600
	s_addc_u32 s1, s1, s7                                      // 00000000175C: 82010701
	s_load_b64 s[6:7], s[0:1], null                            // 000000001760: F4040180 F8000000
	s_clause 0x1                                               // 000000001768: BF850001
	s_load_b64 s[10:11], s[8:9], null                          // 00000000176C: F4040284 F8000000
	s_load_b32 s8, s[8:9], 0x8                                 // 000000001774: F4000204 F8000008
	s_load_b32 s9, s[0:1], 0x8                                 // 00000000177C: F4000240 F8000008
	s_mul_i32 s0, s15, 55                                      // 000000001784: 9600B70F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001788: BF870499
	s_ashr_i32 s1, s0, 31                                      // 00000000178C: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001790: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 000000001794: BF8700D9
	s_add_u32 s4, s4, s0                                       // 000000001798: 80040004
	s_addc_u32 s5, s5, s1                                      // 00000000179C: 82050105
	s_waitcnt lgkmcnt(0)                                       // 0000000017A0: BF89FC07
	v_fma_f32 v0, s10, s6, 0                                   // 0000000017A4: D6130000 02000C0A
	s_mul_i32 s6, s14, 5                                       // 0000000017AC: 9606850E
	v_fmac_f32_e64 v0, s11, s7                                 // 0000000017B0: D52B0000 00000E0B
	s_ashr_i32 s7, s6, 31                                      // 0000000017B8: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017BC: BF870099
	s_lshl_b64 s[0:1], s[6:7], 2                               // 0000000017C0: 84808206
	v_fmac_f32_e64 v0, s8, s9                                  // 0000000017C4: D52B0000 00001208
	v_mov_b32_e32 v1, 0                                        // 0000000017CC: 7E020280
	s_add_u32 s0, s4, s0                                       // 0000000017D0: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017D4: 82010105
	s_add_u32 s0, s0, s2                                       // 0000000017D8: 80000200
	v_max_f32_e32 v0, 0, v0                                    // 0000000017DC: 20000080
	s_addc_u32 s1, s1, s3                                      // 0000000017E0: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 0000000017E4: DC6A0000 00000001
	s_nop 0                                                    // 0000000017EC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017F0: BFB60003
	s_endpgm                                                   // 0000000017F4: BFB00000

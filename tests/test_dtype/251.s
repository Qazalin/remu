
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4n113>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_mov_b32 s2, s15                                          // 000000001708: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 00000000170C: 86039F0F
	v_mov_b32_e32 v0, 0                                        // 000000001710: 7E000280
	s_lshl_b64 s[8:9], s[2:3], 1                               // 000000001714: 84888102
	s_load_b64 s[0:1], s[0:1], 0x10                            // 000000001718: F4040000 F8000010
	s_waitcnt lgkmcnt(0)                                       // 000000001720: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001724: 80060806
	s_addc_u32 s7, s7, s9                                      // 000000001728: 82070907
	s_lshl_b64 s[2:3], s[2:3], 3                               // 00000000172C: 84828302
	global_load_u16 v1, v0, s[6:7]                             // 000000001730: DC4A0000 01060000
	s_add_u32 s0, s0, s2                                       // 000000001738: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000173C: 82010301
	s_load_b64 s[0:1], s[0:1], null                            // 000000001740: F4040000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001748: BF89FC07
	s_clz_i32_u32 s2, s1                                       // 00000000174C: BE820A01
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001750: BF870499
	s_min_u32 s2, s2, 32                                       // 000000001754: 8982A002
	s_lshl_b64 s[0:1], s[0:1], s2                              // 000000001758: 84800200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000175C: BF870499
	s_min_u32 s0, s0, 1                                        // 000000001760: 89808100
	s_or_b32 s0, s1, s0                                        // 000000001764: 8C000001
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001768: BF870009
	v_cvt_f32_u32_e32 v2, s0                                   // 00000000176C: 7E040C00
	s_sub_i32 s0, 32, s2                                       // 000000001770: 818002A0
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001774: BF870481
	v_ldexp_f32 v2, v2, s0                                     // 000000001778: D71C0002 00000102
	s_add_u32 s0, s4, s8                                       // 000000001780: 80000804
	s_addc_u32 s1, s5, s9                                      // 000000001784: 82010905
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001788: BF8700B1
	v_cvt_f16_f32_e32 v2, v2                                   // 00000000178C: 7E041502
	s_waitcnt vmcnt(0)                                         // 000000001790: BF8903F7
	v_cvt_f16_i16_e32 v1, v1                                   // 000000001794: 7E02A301
	v_mul_f16_e32 v1, v1, v2                                   // 000000001798: 6A020501
	global_store_b16 v0, v1, s[0:1]                            // 00000000179C: DC660000 00000100
	s_nop 0                                                    // 0000000017A4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017A8: BFB60003
	s_endpgm                                                   // 0000000017AC: BFB00000

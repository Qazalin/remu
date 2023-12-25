
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_256_320n1>:
	v_mov_b32_e32 v0, 0                                        // 000000001600: 7E000280
	s_mov_b32 s2, s15                                          // 000000001604: BE82000F
	s_mov_b32 s3, 0                                            // 000000001608: BE830080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000160C: BF870009
	s_cmpk_lt_u32 s3, 0x100                                    // 000000001610: B6830100
	s_cselect_b32 s4, -1, 0                                    // 000000001614: 980480C1
	s_cmpk_lt_u32 s3, 0xff                                     // 000000001618: B68300FF
	v_cndmask_b32_e64 v1, 0, 1.0, s4                           // 00000000161C: D5010001 0011E480
	s_cselect_b32 s4, -1, 0                                    // 000000001624: 980480C1
	s_cmpk_lt_u32 s3, 0xfe                                     // 000000001628: B68300FE
	v_cndmask_b32_e64 v2, 0, 1.0, s4                           // 00000000162C: D5010002 0011E480
	s_cselect_b32 s4, -1, 0                                    // 000000001634: 980480C1
	v_add_f32_e32 v0, v0, v1                                   // 000000001638: 06000300
	v_cndmask_b32_e64 v1, 0, 1.0, s4                           // 00000000163C: D5010001 0011E480
	s_cmpk_lt_u32 s3, 0xfd                                     // 000000001644: B68300FD
	s_cselect_b32 s4, -1, 0                                    // 000000001648: 980480C1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 00000000164C: BF870142
	v_add_f32_e32 v0, v0, v2                                   // 000000001650: 06000500
	v_cndmask_b32_e64 v2, 0, 1.0, s4                           // 000000001654: D5010002 0011E480
	s_cmpk_lt_u32 s3, 0xfc                                     // 00000000165C: B68300FC
	s_cselect_b32 s4, -1, 0                                    // 000000001660: 980480C1
	v_add_f32_e32 v0, v0, v1                                   // 000000001664: 06000300
	v_cndmask_b32_e64 v1, 0, 1.0, s4                           // 000000001668: D5010001 0011E480
	s_cmpk_lt_u32 s3, 0xfb                                     // 000000001670: B68300FB
	s_cselect_b32 s4, -1, 0                                    // 000000001674: 980480C1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001678: BF870142
	v_add_f32_e32 v0, v0, v2                                   // 00000000167C: 06000500
	v_cndmask_b32_e64 v2, 0, 1.0, s4                           // 000000001680: D5010002 0011E480
	s_cmpk_lt_u32 s3, 0xfa                                     // 000000001688: B68300FA
	s_cselect_b32 s4, -1, 0                                    // 00000000168C: 980480C1
	v_add_f32_e32 v0, v0, v1                                   // 000000001690: 06000300
	v_cndmask_b32_e64 v1, 0, 1.0, s4                           // 000000001694: D5010001 0011E480
	s_cmpk_lt_u32 s3, 0xf9                                     // 00000000169C: B68300F9
	s_cselect_b32 s4, -1, 0                                    // 0000000016A0: 980480C1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 0000000016A4: BF870142
	v_add_f32_e32 v0, v0, v2                                   // 0000000016A8: 06000500
	v_cndmask_b32_e64 v2, 0, 1.0, s4                           // 0000000016AC: D5010002 0011E480
	s_cmpk_lt_u32 s3, 0xf8                                     // 0000000016B4: B68300F8
	s_cselect_b32 s4, -1, 0                                    // 0000000016B8: 980480C1
	v_add_f32_e32 v0, v0, v1                                   // 0000000016BC: 06000300
	v_cndmask_b32_e64 v1, 0, 1.0, s4                           // 0000000016C0: D5010001 0011E480
	s_cmpk_lt_u32 s3, 0xf7                                     // 0000000016C8: B68300F7
	s_cselect_b32 s4, -1, 0                                    // 0000000016CC: 980480C1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 0000000016D0: BF870142
	v_add_f32_e32 v0, v0, v2                                   // 0000000016D4: 06000500
	v_cndmask_b32_e64 v2, 0, 1.0, s4                           // 0000000016D8: D5010002 0011E480
	s_cmpk_lt_u32 s3, 0xf6                                     // 0000000016E0: B68300F6
	s_cselect_b32 s4, -1, 0                                    // 0000000016E4: 980480C1
	v_add_f32_e32 v0, v0, v1                                   // 0000000016E8: 06000300
	v_cndmask_b32_e64 v1, 0, 1.0, s4                           // 0000000016EC: D5010001 0011E480
	s_cmpk_lt_u32 s3, 0xf5                                     // 0000000016F4: B68300F5
	s_cselect_b32 s4, -1, 0                                    // 0000000016F8: 980480C1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 0000000016FC: BF870142
	v_add_f32_e32 v0, v0, v2                                   // 000000001700: 06000500
	v_cndmask_b32_e64 v2, 0, 1.0, s4                           // 000000001704: D5010002 0011E480
	s_cmpk_lt_u32 s3, 0xf4                                     // 00000000170C: B68300F4
	s_cselect_b32 s4, -1, 0                                    // 000000001710: 980480C1
	v_add_f32_e32 v0, v0, v1                                   // 000000001714: 06000300
	v_cndmask_b32_e64 v1, 0, 1.0, s4                           // 000000001718: D5010001 0011E480
	s_cmpk_lt_u32 s3, 0xf3                                     // 000000001720: B68300F3
	s_cselect_b32 s4, -1, 0                                    // 000000001724: 980480C1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001728: BF870142
	v_add_f32_e32 v0, v0, v2                                   // 00000000172C: 06000500
	v_cndmask_b32_e64 v2, 0, 1.0, s4                           // 000000001730: D5010002 0011E480
	s_cmpk_lt_u32 s3, 0xf2                                     // 000000001738: B68300F2
	s_cselect_b32 s4, -1, 0                                    // 00000000173C: 980480C1
	v_add_f32_e32 v0, v0, v1                                   // 000000001740: 06000300
	v_cndmask_b32_e64 v1, 0, 1.0, s4                           // 000000001744: D5010001 0011E480
	s_cmpk_lt_u32 s3, 0xf1                                     // 00000000174C: B68300F1
	s_cselect_b32 s4, -1, 0                                    // 000000001750: 980480C1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001754: BF870142
	v_add_f32_e32 v0, v0, v2                                   // 000000001758: 06000500
	v_cndmask_b32_e64 v2, 0, 1.0, s4                           // 00000000175C: D5010002 0011E480
	s_cmpk_lt_u32 s3, 0xf0                                     // 000000001764: B68300F0
	s_cselect_b32 s4, -1, 0                                    // 000000001768: 980480C1
	v_add_f32_e32 v0, v0, v1                                   // 00000000176C: 06000300
	v_cndmask_b32_e64 v1, 0, 1.0, s4                           // 000000001770: D5010001 0011E480
	s_cmpk_lt_u32 s3, 0xef                                     // 000000001778: B68300EF
	s_cselect_b32 s4, -1, 0                                    // 00000000177C: 980480C1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001780: BF870142
	v_add_f32_e32 v0, v0, v2                                   // 000000001784: 06000500
	v_cndmask_b32_e64 v2, 0, 1.0, s4                           // 000000001788: D5010002 0011E480
	s_cmpk_lt_u32 s3, 0xee                                     // 000000001790: B68300EE
	s_cselect_b32 s4, -1, 0                                    // 000000001794: 980480C1
	v_add_f32_e32 v0, v0, v1                                   // 000000001798: 06000300
	v_cndmask_b32_e64 v1, 0, 1.0, s4                           // 00000000179C: D5010001 0011E480
	s_cmpk_lt_u32 s3, 0xed                                     // 0000000017A4: B68300ED
	s_cselect_b32 s4, -1, 0                                    // 0000000017A8: 980480C1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000017AC: BF8704B2
	v_add_f32_e32 v0, v0, v2                                   // 0000000017B0: 06000500
	v_cndmask_b32_e64 v2, 0, 1.0, s4                           // 0000000017B4: D5010002 0011E480
	s_add_i32 s3, s3, 20                                       // 0000000017BC: 81039403
	s_cmpk_eq_i32 s3, 0x140                                    // 0000000017C0: B1830140
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017C4: BF870092
	v_add_f32_e32 v0, v0, v1                                   // 0000000017C8: 06000300
	v_add_f32_e32 v0, v0, v2                                   // 0000000017CC: 06000500
	s_cbranch_scc0 65422                                       // 0000000017D0: BFA1FF8E <r_256_320n1+0xc>
	s_load_b64 s[0:1], s[0:1], null                            // 0000000017D4: F4040000 F8000000
	s_ashr_i32 s3, s2, 31                                      // 0000000017DC: 86039F02
	v_mov_b32_e32 v1, 0                                        // 0000000017E0: 7E020280
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000017E4: 84828202
	s_waitcnt lgkmcnt(0)                                       // 0000000017E8: BF89FC07
	s_add_u32 s0, s0, s2                                       // 0000000017EC: 80000200
	s_addc_u32 s1, s1, s3                                      // 0000000017F0: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 0000000017F4: DC6A0000 00000001
	s_nop 0                                                    // 0000000017FC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001800: BFB60003
	s_endpgm                                                   // 000000001804: BFB00000
